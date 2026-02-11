"""
KMZ Mask Generator for remote sensing image mask generation.
Converts KML/KMZ geographic data to binary masks aligned with GeoTIFF images.
"""
import os
import re
import tempfile
import zipfile
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import rasterio
from rasterio.transform import rowcol


class KMZMaskGenerator:
    """
    Generate binary masks from KML/KMZ files aligned with GeoTIFF reference images.
    
    Attributes:
        kmz_path: Path to KML/KMZ file containing geographic features
        tif_path: Path to reference GeoTIFF file for spatial alignment
        target_keywords: List of keywords to match placemark names
        point_radius_pixel: Radius in pixels for point feature expansion
        output_base: Base path for saving output files (optional)
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
            kmz_path: Path to KML/KMZ file
            tif_path: Path to reference GeoTIFF
            target_keywords: Keywords to filter placemarks (default: ['矿体投影', 'Object ID', 'ZK'])
            point_radius_pixel: Radius for point expansion (default: 3)
        """
        self.kmz_path = kmz_path
        self.tif_path = tif_path
        self.target_keywords = target_keywords or ['矿体投影', 'Object ID', 'ZK']
        self.point_radius_pixel = point_radius_pixel
        self.output_base: Optional[str] = None
        
        # Internal state variables
        self.tif_transform: Optional[Any] = None
        self.tif_size: Tuple[int, int] = (0, 0)  # (height, width)
        self.tif_bounds: Dict[str, float] = {}
        self.is_geographic: bool = False
        self.mask: Optional[np.ndarray] = None
        self.geo_data: List[Dict[str, Any]] = []
        
    def extract_and_parse_kml(self) -> 'KMZMaskGenerator':
        """
        Extract KML from KMZ archive (if needed) and parse geographic features.
        
        Returns:
            Self for method chaining
        """
        print('\n🔄 正在读取 KML 数据...')
        _, ext = os.path.splitext(self.kmz_path)
        kml_content = ''
        
        try:
            if ext.lower() in ['.kmz', '.ovkmz']:
                # Extract KML from KMZ archive
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(self.kmz_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find KML file in extracted directory
                    kml_files = [f for f in os.listdir(temp_dir) if f.endswith('.kml')]
                    if not kml_files:
                        raise ValueError('KMZ中未找到KML文件')
                    
                    with open(os.path.join(temp_dir, kml_files[0]), 'r', encoding='utf-8') as f:
                        kml_content = f.read()
            else:
                # Direct KML file
                with open(self.kmz_path, 'r', encoding='utf-8') as f:
                    kml_content = f.read()
        except Exception as e:
            raise RuntimeError(f'读取KML文件失败: {str(e)}')
        
        print('🔍 解析 KML...')
        self._parse_kml_content(kml_content)
        print(f'   已提取 {len(self.geo_data)} 个目标区域')
        
        return self
    
    def _parse_kml_content(self, kml_content: str) -> None:
        """
        Parse KML content and extract matching placemarks.
        
        Args:
            kml_content: Raw KML XML content
        """
        # Use regex to extract placemarks (more robust than XML parsing for malformed KML)
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
            if not any(keyword in area_name for keyword in self.target_keywords):
                continue
            
            # Extract coordinates
            coords_matches = re.findall(r'<coordinates>(.*?)</coordinates>', pm_content)
            if not coords_matches:
                continue
            
            for coords_str in coords_matches:
                coords = self._parse_coordinates(coords_str)
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
                        'coords': coords[0]  # Single point
                    })
                    count += 1
                    print(f'  ✅ 匹配(单点): {area_name}')
    
    def _parse_coordinates(self, coords_str: str) -> Optional[np.ndarray]:
        """
        Parse coordinate string from KML.
        
        Args:
            coords_str: Raw coordinate string (lon,lat,alt or lon,lat)
            
        Returns:
            Nx2 array of (lon, lat) coordinates, or None if parsing fails
        """
        try:
            # Replace commas with spaces and parse numbers
            coords_str = coords_str.strip().replace(',', ' ')
            numbers = [float(x) for x in coords_str.split()]
            
            if len(numbers) == 0:
                return None
            
            # Handle 3D coordinates (lon,lat,alt)
            if len(numbers) % 3 == 0:
                coords = np.array(numbers).reshape(-1, 3)[:, :2]  # Drop altitude
            # Handle 2D coordinates (lon,lat)
            elif len(numbers) % 2 == 0:
                coords = np.array(numbers).reshape(-1, 2)
            else:
                return None
            
            return coords
        except Exception:
            return None
    
    def read_tiff_info(self) -> 'KMZMaskGenerator':
        """
        Read geographic information from reference GeoTIFF.
        
        Returns:
            Self for method chaining
        """
        print('\n🖼️ 读取 TIFF 地理信息...')
        
        try:
            with rasterio.open(self.tif_path) as src:
                self.tif_transform = src.transform
                self.tif_size = (src.height, src.width)
                bounds = src.bounds
                self.tif_bounds = {
                    'xMin': bounds.left,
                    'xMax': bounds.right,
                    'yMin': bounds.bottom,
                    'yMax': bounds.top
                }
                
                # Check if coordinate system is geographic (lat/lon)
                if src.crs and src.crs.is_geographic:
                    self.is_geographic = True
                
        except Exception as e:
            raise RuntimeError(f'读取TIFF信息失败: {str(e)}')
        
        print(f'   尺寸: {self.tif_size[0]} x {self.tif_size[1]}')
        print(f'   X/Lon 范围: {self.tif_bounds["xMin"]:.6f} ~ {self.tif_bounds["xMax"]:.6f}')
        print(f'   Y/Lat 范围: {self.tif_bounds["yMin"]:.6f} ~ {self.tif_bounds["yMax"]:.6f}')
        
        return self
    
    def generate_mask(self) -> 'KMZMaskGenerator':
        """
        Generate binary mask from parsed geographic features.
        
        Returns:
            Self for method chaining
        """
        print('\n🎨 生成蒙版...')
        height, width = self.tif_size
        self.mask = np.zeros((height, width), dtype=bool)
        
        for item in self.geo_data:
            if item['type'] == 'Polygon':
                self._process_polygon(item, height, width)
            elif item['type'] == 'Point':
                self._process_point(item, height, width)
        
        return self
    
    def _process_polygon(self, item: Dict[str, Any], height: int, width: int) -> None:
        """
        Process polygon feature and add to mask.
        
        Args:
            item: Polygon feature data
            height: Mask height
            width: Mask width
        """
        coords = item['coords']
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Convert world coordinates to pixel coordinates
        try:
            rows, cols = rowcol(self.tif_transform, x, y)
            rows = np.array(rows)
            cols = np.array(cols)
        except Exception:
            # Fallback: manual calculation
            pixel_width = (self.tif_bounds['xMax'] - self.tif_bounds['xMin']) / width
            pixel_height = (self.tif_bounds['yMax'] - self.tif_bounds['yMin']) / height
            cols = np.ceil((x - self.tif_bounds['xMin']) / pixel_width).astype(int)
            rows = np.ceil((self.tif_bounds['yMax'] - y) / pixel_height).astype(int)
        
        # Filter valid coordinates
        valid_idx = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        
        if not np.any(valid_idx):
            return
        
        # Close the polygon if needed
        if rows[0] != rows[-1] or cols[0] != cols[-1]:
            rows = np.append(rows, rows[0])
            cols = np.append(cols, cols[0])
        
        # Fill polygon using scanline algorithm
        try:
            # Get bounding box
            min_row = max(0, int(np.floor(rows.min())))
            max_row = min(height, int(np.ceil(rows.max())) + 1)
            min_col = max(0, int(np.floor(cols.min())))
            max_col = min(width, int(np.ceil(cols.max())) + 1)
            
            # Use point-in-polygon test for each pixel in bounding box
            for r in range(min_row, max_row):
                for c in range(min_col, max_col):
                    if self._point_in_polygon(c, r, cols, rows):
                        self.mask[r, c] = True
        except Exception:
            pass
    
    def _point_in_polygon(self, x: float, y: float, poly_x: np.ndarray, poly_y: np.ndarray) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.
        
        Args:
            x, y: Point coordinates
            poly_x, poly_y: Polygon vertex coordinates
            
        Returns:
            True if point is inside polygon
        """
        n = len(poly_x)
        inside = False
        
        p1x, p1y = poly_x[0], poly_y[0]
        for i in range(1, n + 1):
            p2x, p2y = poly_x[i % n], poly_y[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _process_point(self, item: Dict[str, Any], height: int, width: int) -> None:
        """
        Process point feature and add to mask with expansion.
        
        Args:
            item: Point feature data
            height: Mask height
            width: Mask width
        """
        coord = item['coords']
        x, y = coord[0], coord[1]
        
        # Convert world coordinates to pixel coordinates
        try:
            row, col = rowcol(self.tif_transform, [x], [y])
            row, col = row[0], col[0]
        except Exception:
            # Fallback: manual calculation
            pixel_width = (self.tif_bounds['xMax'] - self.tif_bounds['xMin']) / width
            pixel_height = (self.tif_bounds['yMax'] - self.tif_bounds['yMin']) / height
            col = int(np.ceil((x - self.tif_bounds['xMin']) / pixel_width))
            row = int(np.ceil((self.tif_bounds['yMax'] - y) / pixel_height))
        
        # Check if point is within bounds
        if not (0 <= row < height and 0 <= col < width):
            return
        
        # Create circular expansion around point
        rad = self.point_radius_pixel
        y_grid, x_grid = np.ogrid[-rad:rad+1, -rad:rad+1]
        disk = (x_grid**2 + y_grid**2) <= rad**2
        
        # Calculate bounds with clipping
        r_min = max(0, row - rad)
        r_max = min(height, row + rad + 1)
        c_min = max(0, col - rad)
        c_max = min(width, col + rad + 1)
        
        # Calculate disk slice
        dr_min = r_min - (row - rad)
        dr_max = dr_min + (r_max - r_min)
        dc_min = c_min - (col - rad)
        dc_max = dc_min + (c_max - c_min)
        
        # Apply disk to mask
        self.mask[r_min:r_max, c_min:c_max] |= disk[dr_min:dr_max, dc_min:dc_max]
    
    def save_results(self, output_base: str) -> None:
        """
        Save mask results to files.
        
        Args:
            output_base: Base path for output files (without extension)
        """
        print('\n💾 保存结果...')
        
        if self.mask is None:
            raise RuntimeError('Mask not generated yet. Call run() first.')
        
        # Create output directory if needed
        out_dir = os.path.dirname(output_base)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Save as numpy file
        np.save(f'{output_base}.npy', self.mask)
        
        # Save visual check as PNG using scipy or opencv if available
        try:
            import cv2
            cv2.imwrite(f'{output_base}_VisualCheck.png', (self.mask * 255).astype(np.uint8))
            print(f'   保存完毕: {output_base}_VisualCheck.png')
        except ImportError:
            try:
                from scipy.misc import imsave
                imsave(f'{output_base}_VisualCheck.png', self.mask)
                print(f'   保存完毕: {output_base}_VisualCheck.png')
            except (ImportError, AttributeError):
                # Just save numpy array if no image library available
                print(f'   保存完毕: {output_base}.npy (图像保存需要 opencv-python)')
    
    def run(self) -> np.ndarray:
        """
        Execute complete mask generation pipeline.
        
        Returns:
            Binary mask as boolean numpy array
        """
        self.extract_and_parse_kml()
        self.read_tiff_info()
        self.generate_mask()
        
        if self.output_base:
            self.save_results(self.output_base)
        
        print('\n🎉 完成!')
        return self.mask
