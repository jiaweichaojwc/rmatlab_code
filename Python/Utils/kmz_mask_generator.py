"""
KMZMaskGenerator - Generate mask from KML/KMZ files
Converted from MATLAB KMZMaskGenerator.m
"""

import numpy as np
import os
import zipfile
import tempfile
import re
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Polygon, Point
from skimage.draw import polygon as draw_polygon


class KMZMaskGenerator:
    """
    Remote sensing image mask generator from KML/KMZ files
    """
    
    def __init__(self, kmz_path, tif_path, target_keywords=None, point_radius_pixel=3):
        self.kmz_path = kmz_path
        self.tif_path = tif_path
        self.target_keywords = target_keywords or ['矿体投影', 'Object ID', 'ZK']
        self.point_radius_pixel = point_radius_pixel
        self.output_base = None
        
        # Internal state variables
        self.tif_geo_info = None
        self.tif_size = None
        self.tif_limits = None
        self.is_geographic = False
        self.mask = None
        self.geo_data = []
    
    def extract_and_parse_kml(self):
        """Extract and parse KML data"""
        print('\n🔄 正在读取 KML 数据...')
        
        _, ext = os.path.splitext(self.kmz_path)
        kml_content = ''
        temp_dir = None
        
        try:
            if ext.lower() in ['.kmz', '.ovkmz']:
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(self.kmz_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find KML file in extracted files
                kml_files = [f for f in os.listdir(temp_dir) if f.endswith('.kml')]
                if not kml_files:
                    raise ValueError('KMZ中未找到KML文件')
                
                with open(os.path.join(temp_dir, kml_files[0]), 'r', encoding='utf-8') as f:
                    kml_content = f.read()
            else:
                with open(self.kmz_path, 'r', encoding='utf-8') as f:
                    kml_content = f.read()
        finally:
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        
        print('🔍 解析 KML...')
        
        # Parse Placemarks
        placemark_pattern = r'<Placemark>(.*?)</Placemark>'
        placemarks = re.findall(placemark_pattern, kml_content, re.DOTALL)
        
        count = 0
        for pm_content in placemarks:
            # Extract name
            name_match = re.search(r'<name>(.*?)</name>', pm_content)
            if not name_match:
                continue
            area_name = name_match.group(1).strip()
            
            # Check if name matches any keyword
            is_match = any(keyword in area_name for keyword in self.target_keywords)
            if not is_match:
                continue
            
            # Extract coordinates
            coords_matches = re.findall(r'<coordinates>(.*?)</coordinates>', pm_content, re.DOTALL)
            if not coords_matches:
                continue
            
            for coords_text in coords_matches:
                # Parse coordinate string
                coords_text = coords_text.strip().replace(',', ' ')
                raw_data = [float(x) for x in coords_text.split() if x]
                
                if len(raw_data) == 0:
                    continue
                
                # Reshape coordinates
                if len(raw_data) % 3 == 0:
                    coords = np.array(raw_data).reshape(-1, 3)[:, :2]
                elif len(raw_data) % 2 == 0:
                    coords = np.array(raw_data).reshape(-1, 2)
                else:
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
                        'coords': coords[0]
                    })
                    count += 1
                    print(f'  ✅ 匹配(单点): {area_name}')
        
        print(f'   已提取 {count} 个目标区域')
        return self
    
    def read_tiff_info(self):
        """Read TIFF geographic information"""
        print('\n🖼️ 读取 TIFF 地理信息...')
        
        with rasterio.open(self.tif_path) as src:
            self.tif_geo_info = src.transform
            self.tif_size = (src.height, src.width)
            bounds = src.bounds
            self.is_geographic = src.crs.is_geographic if src.crs else False
        
        self.tif_limits = {
            'xMin': bounds.left,
            'xMax': bounds.right,
            'yMin': bounds.bottom,
            'yMax': bounds.top
        }
        
        print(f'   尺寸: {self.tif_size[0]} x {self.tif_size[1]}')
        print(f'   X/Lon 范围: {self.tif_limits["xMin"]:.6f} ~ {self.tif_limits["xMax"]:.6f}')
        print(f'   Y/Lat 范围: {self.tif_limits["yMin"]:.6f} ~ {self.tif_limits["yMax"]:.6f}')
        
        return self
    
    def generate_mask(self):
        """Generate mask from geographic data"""
        print('\n🎨 生成蒙版...')
        
        height, width = self.tif_size
        self.mask = np.zeros((height, width), dtype=bool)
        
        transform = self.tif_geo_info
        pixel_width = (self.tif_limits['xMax'] - self.tif_limits['xMin']) / width
        pixel_height = (self.tif_limits['yMax'] - self.tif_limits['yMin']) / height
        
        for item in self.geo_data:
            coords = item['coords']
            
            if item['type'] == 'Polygon':
                x = coords[:, 0]
                y = coords[:, 1]
                
                # Convert geographic to pixel coordinates
                try:
                    rows, cols = rowcol(transform, x, y)
                except:
                    # Manual calculation fallback
                    cols = ((x - self.tif_limits['xMin']) / pixel_width).astype(int)
                    rows = ((self.tif_limits['yMax'] - y) / pixel_height).astype(int)
                
                # Filter valid coordinates
                valid_idx = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                rows = rows[valid_idx]
                cols = cols[valid_idx]
                
                if len(rows) > 0:
                    # Close polygon if needed
                    if rows[0] != rows[-1] or cols[0] != cols[-1]:
                        rows = np.append(rows, rows[0])
                        cols = np.append(cols, cols[0])
                    
                    # Draw polygon
                    rr, cc = draw_polygon(rows, cols, (height, width))
                    self.mask[rr, cc] = True
            
            elif item['type'] == 'Point':
                x, y = coords[0], coords[1] if len(coords) > 1 else coords[0]
                
                # Convert to pixel coordinates
                try:
                    rows, cols = rowcol(transform, [x], [y])
                    r, c = rows[0], cols[0]
                except:
                    c = int((x - self.tif_limits['xMin']) / pixel_width)
                    r = int((self.tif_limits['yMax'] - y) / pixel_height)
                
                # Draw circle around point
                if 0 <= r < height and 0 <= c < width:
                    rad = self.point_radius_pixel
                    y_coords, x_coords = np.ogrid[-rad:rad+1, -rad:rad+1]
                    disk = (x_coords**2 + y_coords**2) <= rad**2
                    
                    r_min = max(0, r - rad)
                    r_max = min(height, r + rad + 1)
                    c_min = max(0, c - rad)
                    c_max = min(width, c + rad + 1)
                    
                    # Adjust disk size to fit within bounds
                    dr_min = rad - (r - r_min)
                    dr_max = dr_min + (r_max - r_min)
                    dc_min = rad - (c - c_min)
                    dc_max = dc_min + (c_max - c_min)
                    
                    self.mask[r_min:r_max, c_min:c_max] |= disk[dr_min:dr_max, dc_min:dc_max]
        
        return self
    
    def save_results(self, output_base):
        """Save results to files"""
        print('\n💾 保存结果...')
        
        final_mask = self.mask
        
        # Create output directory if needed
        out_dir = os.path.dirname(output_base)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Save as .npy (Python equivalent of .mat)
        np.save(output_base + '.npy', final_mask)
        
        # Save visualization
        from PIL import Image
        img = Image.fromarray((final_mask * 255).astype(np.uint8))
        img.save(output_base + '_VisualCheck.png')
        
        print(f'   保存完毕: {output_base}_VisualCheck.png')
    
    def run(self):
        """Main execution pipeline"""
        self.extract_and_parse_kml()
        self.read_tiff_info()
        self.generate_mask()
        
        if self.output_base:
            self.save_results(self.output_base)
        
        print('\n🎉 完成!')
        return self.mask
