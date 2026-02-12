"""
KMZMaskGenerator - KML/KMZ parsing and rasterization for mask generation.

This module provides tools for:
- Extracting and parsing KML/KMZ files
- Converting geographic coordinates to raster pixels
- Generating binary masks from polygons and points
- Aligning with reference TIFF geospatial information
"""

import os
import zipfile
import tempfile
import shutil
import re
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, Point, mapping
from shapely.ops import transform
import fiona
from fiona.crs import from_epsg
import warnings
from scipy.io import savemat
try:
    import xml.etree.ElementTree as ET
except ImportError:
    print("Warning: xml.etree not available")

warnings.filterwarnings('ignore')


class KMZMaskGenerator:
    """
    KML/KMZ mask generator for remote sensing images.
    
    This class handles extraction and parsing of KML/KMZ files, then generates
    binary masks by rasterizing the geometries onto a reference TIFF grid.
    """
    
    def __init__(
        self,
        kmz_path: str,
        tif_path: str,
        target_keywords: Optional[List[str]] = None,
        point_radius_pixel: int = 3
    ):
        """
        Initialize KMZMaskGenerator.
        
        Args:
            kmz_path: Path to KML or KMZ file
            tif_path: Path to reference TIFF file
            target_keywords: List of keywords to filter features (default: ['矿体投影', 'Object ID', 'ZK'])
            point_radius_pixel: Radius in pixels for point features (default: 3)
        """
        self.kmz_path = kmz_path
        self.tif_path = tif_path
        self.target_keywords = target_keywords or ['矿体投影', 'Object ID', 'ZK']
        self.point_radius_pixel = point_radius_pixel
        self.output_base: Optional[str] = None
        
        # Internal state variables
        self.tif_geo_info: Optional[rasterio.Affine] = None
        self.tif_size: Optional[Tuple[int, int]] = None  # (height, width)
        self.tif_limits: Optional[Dict[str, float]] = None
        self.is_geographic: bool = False
        self.mask: Optional[np.ndarray] = None
        self.geo_data: List[Dict[str, Any]] = []
    
    def extract_and_parse_kml(self) -> None:
        """Extract KML from KMZ (if needed) and parse geometric features."""
        print('\n🔄 正在读取 KML 数据...')
        
        _, ext = os.path.splitext(self.kmz_path)
        kml_content = ''
        temp_dir = None
        
        try:
            if ext.lower() in ['.kmz', '.ovkmz']:
                # Extract KMZ to temporary directory
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(self.kmz_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find KML file
                kml_files = [f for f in os.listdir(temp_dir) if f.endswith('.kml')]
                if not kml_files:
                    raise ValueError('KMZ中未找到KML文件')
                
                kml_path = os.path.join(temp_dir, kml_files[0])
                with open(kml_path, 'r', encoding='utf-8') as f:
                    kml_content = f.read()
            else:
                # Direct KML file
                with open(self.kmz_path, 'r', encoding='utf-8') as f:
                    kml_content = f.read()
        finally:
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        print('🔍 解析 KML...')
        self._parse_kml_content(kml_content)
    
    def _parse_kml_content(self, kml_content: str) -> None:
        """
        Parse KML content and extract geometries matching target keywords.
        
        Args:
            kml_content: KML file content as string
        """
        # Use regex to extract Placemarks
        placemark_pattern = r'(?s)<Placemark>(.*?)</Placemark>'
        placemarks = re.findall(placemark_pattern, kml_content)
        
        count = 0
        for pm_content in placemarks:
            # Extract name
            name_match = re.search(r'<name>(.*?)</name>', pm_content)
            if not name_match:
                continue
            
            area_name = name_match.group(1).strip()
            
            # Check if name matches any target keyword
            is_match = any(keyword in area_name for keyword in self.target_keywords)
            if not is_match:
                continue
            
            # Extract coordinates
            coords_matches = re.findall(r'<coordinates>(.*?)</coordinates>', pm_content)
            if not coords_matches:
                continue
            
            for coords_text in coords_matches:
                coords = self._parse_coordinates(coords_text)
                if coords is None or len(coords) == 0:
                    continue
                
                # Determine geometry type
                if '<Polygon>' in pm_content or '<LinearRing>' in pm_content:
                    self.geo_data.append({
                        'type': 'Polygon',
                        'name': area_name,
                        'coords': coords
                    })
                    count += 1
                    print(f'  ✅ 匹配(多边形): {area_name}')
                elif '<Point>' in pm_content:
                    self.geo_data.append({
                        'type': 'Point',
                        'name': area_name,
                        'coords': coords[0] if len(coords) > 0 else coords
                    })
                    count += 1
                    print(f'  ✅ 匹配(单点): {area_name}')
        
        print(f'   已提取 {count} 个目标区域')
    
    def _parse_coordinates(self, coords_text: str) -> Optional[np.ndarray]:
        """
        Parse coordinate string from KML.
        
        Args:
            coords_text: Coordinate string from KML
            
        Returns:
            Array of shape (N, 2) with [lon, lat] coordinates
        """
        # Replace commas with spaces for easier parsing
        coords_text = coords_text.strip().replace(',', ' ')
        
        try:
            # Parse all numbers
            raw_data = [float(x) for x in coords_text.split()]
            
            if len(raw_data) == 0:
                return None
            
            # KML coordinates are: lon,lat,alt or lon,lat
            if len(raw_data) % 3 == 0:
                # Has altitude
                coords = np.array(raw_data).reshape(-1, 3)[:, :2]  # Take only lon, lat
            elif len(raw_data) % 2 == 0:
                # Only lon, lat
                coords = np.array(raw_data).reshape(-1, 2)
            else:
                return None
            
            return coords
            
        except (ValueError, TypeError):
            return None
    
    def read_tiff_info(self) -> None:
        """Read geospatial information from reference TIFF file."""
        print('\n🖼️ 读取 TIFF 地理信息...')
        
        with rasterio.open(self.tif_path) as src:
            self.tif_geo_info = src.transform
            self.tif_size = (src.height, src.width)
            self.is_geographic = src.crs.is_geographic if src.crs else False
            
            # Get bounds
            bounds = src.bounds
            self.tif_limits = {
                'xMin': bounds.left,
                'xMax': bounds.right,
                'yMin': bounds.bottom,
                'yMax': bounds.top
            }
        
        print(f'   尺寸: {self.tif_size[0]} x {self.tif_size[1]}')
        print(f'   X/Lon 范围: {self.tif_limits["xMin"]:.6f} ~ {self.tif_limits["xMax"]:.6f}')
        print(f'   Y/Lat 范围: {self.tif_limits["yMin"]:.6f} ~ {self.tif_limits["yMax"]:.6f}')
    
    def _world_to_pixel(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            x: X or longitude coordinates
            y: Y or latitude coordinates
            
        Returns:
            Tuple of (rows, cols) in pixel coordinates
        """
        # Use rasterio transform
        inv_transform = ~self.tif_geo_info
        cols, rows = [], []
        
        for xi, yi in zip(x, y):
            col, row = inv_transform * (xi, yi)
            cols.append(int(col))
            rows.append(int(row))
        
        return np.array(rows), np.array(cols)
    
    def generate_mask(self) -> None:
        """Generate binary mask by rasterizing geometries."""
        print('\n🎨 生成蒙版...')
        
        height, width = self.tif_size
        self.mask = np.zeros((height, width), dtype=bool)
        
        for item in self.geo_data:
            coords = item['coords']
            
            if item['type'] == 'Polygon':
                self._rasterize_polygon(coords)
            elif item['type'] == 'Point':
                self._rasterize_point(coords)
    
    def _rasterize_polygon(self, coords: np.ndarray) -> None:
        """
        Rasterize polygon onto mask.
        
        Args:
            coords: Array of shape (N, 2) with [lon, lat] coordinates
        """
        from skimage.draw import polygon
        
        x = coords[:, 0]
        y = coords[:, 1]
        
        rows, cols = self._world_to_pixel(x, y)
        
        # Ensure polygon is closed
        if rows[0] != rows[-1] or cols[0] != cols[-1]:
            rows = np.append(rows, rows[0])
            cols = np.append(cols, cols[0])
        
        # Filter valid coordinates
        height, width = self.tif_size
        valid_idx = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        
        if np.sum(valid_idx) > 2:
            # Use skimage polygon to get interior pixels
            try:
                rr, cc = polygon(rows[valid_idx], cols[valid_idx], self.mask.shape)
                # Clip to image bounds
                valid_mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                self.mask[rr[valid_mask], cc[valid_mask]] = True
            except Exception as e:
                print(f'Warning: Could not rasterize polygon: {e}')
    
    def _rasterize_point(self, coords: np.ndarray) -> None:
        """
        Rasterize point onto mask with circular buffer.
        
        Args:
            coords: Single coordinate [lon, lat]
        """
        if coords.ndim == 1:
            x, y = coords[0], coords[1]
        else:
            x, y = coords[0, 0], coords[0, 1]
        
        rows, cols = self._world_to_pixel(np.array([x]), np.array([y]))
        r, c = rows[0], cols[0]
        
        height, width = self.tif_size
        
        # Check if point is within image bounds
        if 0 <= c < width and 0 <= r < height:
            rad = self.point_radius_pixel
            
            # Create circular disk
            yy, xx = np.ogrid[-rad:rad+1, -rad:rad+1]
            disk = (xx**2 + yy**2) <= rad**2
            
            # Calculate bounds
            r_min = max(0, r - rad)
            r_max = min(height, r + rad + 1)
            c_min = max(0, c - rad)
            c_max = min(width, c + rad + 1)
            
            # Calculate disk slice
            dr_min = rad - (r - r_min)
            dr_max = dr_min + (r_max - r_min)
            dc_min = rad - (c - c_min)
            dc_max = dc_min + (c_max - c_min)
            
            # Apply disk to mask
            self.mask[r_min:r_max, c_min:c_max] |= disk[dr_min:dr_max, dc_min:dc_max]
    
    def save_results(self, output_base: str) -> None:
        """
        Save mask results to MAT and PNG files.
        
        Args:
            output_base: Base path for output files (without extension)
        """
        print('\n💾 保存结果...')
        
        # Create output directory if needed
        out_dir = os.path.dirname(output_base)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Save as MAT file
        final_mask = self.mask
        savemat(f'{output_base}.mat', {'finalMask': final_mask})
        
        # Save as PNG for visual check
        from PIL import Image
        img = Image.fromarray((self.mask * 255).astype(np.uint8), mode='L')
        img.save(f'{output_base}_VisualCheck.png')
        
        print(f'   保存完毕: {output_base}_VisualCheck.png')
    
    def run(self) -> np.ndarray:
        """
        Execute full KMZ mask generation pipeline.
        
        Returns:
            Binary mask as numpy array
        """
        self.extract_and_parse_kml()
        self.read_tiff_info()
        self.generate_mask()
        
        if self.output_base:
            self.save_results(self.output_base)
        
        print('\n🎉 完成!')
        return self.mask
