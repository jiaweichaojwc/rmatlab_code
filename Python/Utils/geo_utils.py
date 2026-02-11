"""
GeoUtils - Geospatial utility functions
Converted from MATLAB GeoUtils.m
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon, Point
import pandas as pd
import os
from tkinter import Tk, filedialog, messagebox
import warnings


class GeoUtils:
    """Static utility class for geospatial operations"""
    
    @staticmethod
    def getRegionConfig(region_type):
        """
        Interactive mode: force manual selection of data folder and coordinate file
        Returns: data_dir, roi_file, belt_lon, belt_lat
        """
        # Hide the root Tkinter window
        root = Tk()
        root.withdraw()
        
        print('>>> [交互模式] 1. 请选择 data 数据文件夹...')
        data_dir = filedialog.askdirectory(title='请选择 data 数据文件夹')
        if not data_dir:
            raise ValueError('用户取消了文件夹选择，程序终止。')
        
        # Default path from parent directory of data
        default_path = os.path.dirname(data_dir)
        
        print('>>> [交互模式] 2. 请选择经纬度坐标 Excel 文件...')
        roi_file = filedialog.askopenfilename(
            title='请选择经纬度坐标文件',
            initialdir=default_path,
            filetypes=[('Excel files', '*.xlsx *.xls'), ('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not roi_file:
            raise ValueError('用户取消了坐标文件选择，程序终止。')
        
        print(f'    ✅ 已选择坐标文件: {roi_file}')
        
        # Get belt coordinates
        belt_lon, belt_lat = GeoUtils.get_belt_coords(os.path.dirname(roi_file), roi_file)
        
        return data_dir, roi_file, belt_lon, belt_lat
    
    @staticmethod
    def get_belt_coords(root_dir, roi_file):
        """Get belt coordinates from ROI file"""
        if os.path.isfile(roi_file):
            fullpath_roi = roi_file
        else:
            fullpath_roi = os.path.join(root_dir, roi_file)
        
        # Use smart reading to get valid coordinates
        _, _, _, _, lon_valid, lat_valid = GeoUtils.readROI_Robust(fullpath_roi)
        
        # Construct closed rectangular frame
        x1 = round(np.min(lon_valid), 4)
        x2 = round(np.max(lon_valid), 4)
        y1 = round(np.min(lat_valid), 4)
        y2 = round(np.max(lat_valid), 4)
        
        belt_lon = np.array([x1, x2, x2, x1, x1])
        belt_lat = np.array([y1, y1, y2, y2, y1])
        
        print(f'    自动识别坐标范围: Lon[{x1:.2f} ~ {x2:.2f}], Lat[{y1:.2f} ~ {y2:.2f}]')
        
        return belt_lon, belt_lat
    
    @staticmethod
    def readROI_Robust(fullpath_roi):
        """
        Smart ROI reading function
        Returns: roiPoly, inROI_vec, lonGrid, latGrid, lonROI, latROI
        """
        try:
            # Read file based on extension
            ext = os.path.splitext(fullpath_roi)[1].lower()
            if ext == '.csv':
                raw_data = pd.read_csv(fullpath_roi, header=None).values
            else:  # Excel files
                raw_data = pd.read_excel(fullpath_roi, header=None).values
            
            rows, cols = raw_data.shape
            
            # Smart identification of lon/lat columns
            candidates_lon = []
            candidates_lat = []
            parsed_cols = []
            
            for c in range(cols):
                col_data = raw_data[:, c]
                nums = np.full(rows, np.nan)
                valid_count = 0
                
                for r in range(rows):
                    val = col_data[r]
                    try:
                        if pd.notna(val):
                            nums[r] = float(val)
                            valid_count += 1
                    except (ValueError, TypeError):
                        nums[r] = np.nan
                
                if valid_count > 3:
                    valid_nums = nums[~np.isnan(nums)]
                    mean_val = np.mean(valid_nums)
                    min_val = np.min(valid_nums)
                    max_val = np.max(valid_nums)
                    parsed_cols.append(nums)
                    
                    # Feature judgment
                    if 60 < mean_val < 160 and (max_val - min_val) < 20:
                        candidates_lon.append(c)
                    if 0 < mean_val < 60 and (max_val - min_val) < 20:
                        candidates_lat.append(c)
                else:
                    parsed_cols.append(None)
            
            # Decision logic
            if len(candidates_lon) == 1 and len(candidates_lat) == 1:
                final_lon_col = candidates_lon[0]
                final_lat_col = candidates_lat[0]
            else:
                # Default: column 1 (index 1) for Lon, column 2 (index 2) for Lat
                if cols >= 3 and parsed_cols[1] is not None:
                    final_lon_col = 1
                    final_lat_col = 2
                elif cols >= 2 and parsed_cols[0] is not None:
                    final_lon_col = 0
                    final_lat_col = 1
                else:
                    raise ValueError('无法识别经纬度列，请检查文件格式。')
            
            lonROI = parsed_cols[final_lon_col]
            latROI = parsed_cols[final_lat_col]
            
            valid_mask = ~np.isnan(lonROI) & ~np.isnan(latROI)
            lonROI = lonROI[valid_mask]
            latROI = latROI[valid_mask]
            
            if len(lonROI) == 0:
                raise ValueError('没有有效坐标数据')
            
            # Close the polygon
            if abs(lonROI[0] - lonROI[-1]) > 1e-6 or abs(latROI[0] - latROI[-1]) > 1e-6:
                lonROI = np.append(lonROI, lonROI[0])
                latROI = np.append(latROI, latROI[0])
            
            # Create polygon using shapely
            roiPoly = Polygon(zip(lonROI, latROI))
            
            # Empty placeholders for grid data (will be filled later if needed)
            inROI_vec = np.array([])
            lonGrid = np.array([])
            latGrid = np.array([])
            
            return roiPoly, inROI_vec, lonGrid, latGrid, lonROI, latROI
            
        except Exception as e:
            print(f'读取坐标文件出错: {str(e)}')
            raise
    
    @staticmethod
    def readSentinel2(data_dir):
        """Read Sentinel-2 L2A data"""
        import glob
        
        # Find Sentinel-2 directory
        s2_dirs = glob.glob(os.path.join(data_dir, 'Sentinel*2 L2*'))
        if not s2_dirs:
            s2_dirs = glob.glob(os.path.join(data_dir, 'Sentinel*2*'))
        if not s2_dirs:
            raise FileNotFoundError('未找到Sentinel-2 L2A文件')
        
        s2_dir = s2_dirs[0]
        
        # Find B08 file or any tif file
        files = glob.glob(os.path.join(s2_dir, '*B08*'))
        if not files:
            files = glob.glob(os.path.join(s2_dir, '*.tif*'))
        if not files:
            files = glob.glob(os.path.join(s2_dir, '*.jp2'))
        if not files:
            raise FileNotFoundError('未找到Sentinel-2数据文件')
        
        firstFile = files[0]
        ref_tif_path = firstFile
        
        # Read reference for geotransform
        with rasterio.open(firstFile) as src:
            R = src.transform
            height = src.height
            width = src.width
            crs = src.crs
            bounds = src.bounds
        
        # Store reference info in a dict
        R_dict = {
            'transform': R,
            'height': height,
            'width': width,
            'crs': crs,
            'bounds': bounds
        }
        
        # Read bands: B02, B03, B04, B08, B11, B12, B05, B06, B07
        s2_patterns = [['B02'], ['B03'], ['B04'], ['B08'], ['B11'], ['B12'], ['B05'], ['B06'], ['B07']]
        s2_raw = GeoUtils.readMultiBands_smart(s2_dir, s2_patterns, R_dict, 9)
        s2 = s2_raw.astype(np.float32) * 0.0001
        
        return s2, R_dict, ref_tif_path
    
    @staticmethod
    def readLandsat8(data_dir, R):
        """Read Landsat 8 data"""
        import glob
        
        lan_l1 = glob.glob(os.path.join(data_dir, '*Landsat*8*L1*'))
        lan_l2 = glob.glob(os.path.join(data_dir, '*Landsat*8*L2*'))
        
        if lan_l1:
            lan_dir = lan_l1[0]
        elif lan_l2:
            lan_dir = lan_l2[0]
        else:
            raise FileNotFoundError('未找到Landsat 8数据')
        
        lan_patterns = [['B2'], ['B3'], ['B4'], ['B5'], ['B6'], ['B7'], ['B8']]
        lan = GeoUtils.readMultiBands_smart(lan_dir, lan_patterns, R, 7)
        
        return lan
    
    @staticmethod
    def readASTER(data_dir, R):
        """Read ASTER data"""
        import glob
        
        aster_dirs = glob.glob(os.path.join(data_dir, '*ASTER*L2*'))
        if not aster_dirs:
            aster_dirs = glob.glob(os.path.join(data_dir, '*ASTER*L1*'))
        if not aster_dirs:
            raise FileNotFoundError('未找到ASTER数据')
        
        aster_dir = aster_dirs[0]
        aster_pat = [
            ['B01', 'B1'], ['B02', 'B2'], ['B3N', 'B03N'], ['B04', 'B4'],
            ['B05', 'B5'], ['B06', 'B6'], ['B07', 'B7'], ['B08', 'B8'],
            ['B09'], ['B10'], ['B11'], ['B12'], ['B13'], ['B14']
        ]
        
        H, W = R['height'], R['width']
        ast = np.full((H, W, 14), np.nan, dtype=np.float32)
        
        for b in range(14):
            single_band = GeoUtils.readAny_smart(aster_dir, aster_pat[b], R)
            if b < 9:  # VNIR and SWIR bands
                single_band = single_band * 0.01
                single_band[np.isinf(single_band)] = np.nan
            else:  # TIR bands
                single_band = single_band * 0.1 + 300
                single_band[np.isinf(single_band)] = 300
            ast[:, :, b] = single_band
        
        return ast
    
    @staticmethod
    def readDEMandROI(data_dir, roi_file, R):
        """Read DEM and ROI data"""
        import glob
        
        dem_files = glob.glob(os.path.join(data_dir, 'DEM.tif'))
        if not dem_files:
            dem_files = glob.glob(os.path.join(data_dir, 'DEM.tiff'))
        
        H, W = R['height'], R['width']
        bounds = R['bounds']
        
        # Create grid
        lonVec = np.linspace(bounds.left, bounds.right, W)
        latVec = np.linspace(bounds.bottom, bounds.top, H)
        lonGrid, latGrid = np.meshgrid(lonVec, latVec)
        
        # Read ROI file
        if os.path.isfile(roi_file):
            fullpath = roi_file
        else:
            root_dir = os.path.dirname(data_dir)
            fullpath = os.path.join(root_dir, roi_file)
        
        roiPoly, _, _, _, lonROI, latROI = GeoUtils.readROI_Robust(fullpath)
        
        # Create inROI mask
        inROI_vec = np.array([roiPoly.contains(Point(lon, lat)) 
                              for lon, lat in zip(lonGrid.flatten(), latGrid.flatten())])
        inROI = np.flipud(inROI_vec.reshape(H, W))
        
        # Read DEM
        if dem_files:
            with rasterio.open(dem_files[0]) as src:
                dem_raw = src.read(1)
            dem = dem_raw.astype(np.float32)
            dem[np.isinf(dem)] = np.nan
            
            # Resize if needed
            if dem.shape != (H, W):
                from scipy.ndimage import zoom
                zoom_factors = (H / dem.shape[0], W / dem.shape[1])
                dem = zoom(dem, zoom_factors, order=1)
        else:
            dem = np.full((H, W), np.nan, dtype=np.float32)
        
        return dem, inROI, lonGrid, latGrid, lonROI, latROI
    
    @staticmethod
    def readMultiBands_smart(dirPath, patterns_cell, R, numBands):
        """Read multiple bands smartly"""
        import glob
        
        H, W = R['height'], R['width']
        cube = np.full((H, W, numBands), np.nan, dtype=np.float32)
        
        files = glob.glob(os.path.join(dirPath, '*.tif'))
        files += glob.glob(os.path.join(dirPath, '*.tiff'))
        files += glob.glob(os.path.join(dirPath, '*.jp2'))
        
        for b in range(numBands):
            pattern = patterns_cell[b]
            for file_path in files:
                fname = os.path.basename(file_path).upper()
                if any(p.upper() in fname for p in pattern):
                    with rasterio.open(file_path) as src:
                        A = src.read(1).astype(np.float32)
                    
                    # Resize if needed
                    if A.shape != (H, W):
                        from scipy.ndimage import zoom
                        zoom_factors = (H / A.shape[0], W / A.shape[1])
                        A = zoom(A, zoom_factors, order=1)
                    
                    cube[:, :, b] = A
                    break
        
        return cube
    
    @staticmethod
    def readAny_smart(dirPath, patterns_cell, R):
        """Read any single band smartly"""
        if isinstance(patterns_cell, str):
            patterns_cell = [patterns_cell]
        cube = GeoUtils.readMultiBands_smart(dirPath, [patterns_cell], R, 1)
        return cube[:, :, 0]
    
    @staticmethod
    def getBand(*args):
        """Get band from multiple sources, prioritizing non-zero/non-NaN data"""
        idx = args[-1]  # Last argument is the band index
        cubes = args[:-1]  # All other arguments are data cubes
        
        for cube in cubes:
            if cube.shape[2] >= idx:
                band = cube[:, :, idx - 1]  # Convert to 0-indexed
                if np.count_nonzero(~np.isnan(band) & (band != 0)) > 100:
                    return band
        
        # Return NaN array if no valid band found
        H, W = cubes[0].shape[:2]
        return np.full((H, W), np.nan, dtype=np.float32)
    
    @staticmethod
    def mat2gray_roi(img, inROI, min_val=None, max_val=None):
        """Normalize image to [0, 1] within ROI"""
        img_norm = np.full_like(img, np.nan, dtype=np.float32)
        img_roi = img[inROI]
        img_roi = img_roi[~np.isnan(img_roi) & ~np.isinf(img_roi)]
        
        if len(img_roi) == 0:
            return img_norm
        
        if min_val is None:
            min_val = np.min(img_roi)
        if max_val is None:
            max_val = np.max(img_roi)
        
        if max_val - min_val < np.finfo(float).eps:
            img_norm[inROI] = 0.5
        else:
            val = (img[inROI] - min_val) / (max_val - min_val)
            val[val < 0] = 0
            val[val > 1] = 1
            img_norm[inROI] = val
        
        return img_norm
    
    @staticmethod
    def calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets):
        """Calculate S2REP (Red Edge Position) from DN values"""
        B4_val = (B4 * 10000) * scale_factors[0] + offsets[0]
        B5_val = (B5 * 10000) * scale_factors[1] + offsets[1]
        B6_val = (B6 * 10000) * scale_factors[2] + offsets[2]
        B7_val = (B7 * 10000) * scale_factors[3] + offsets[3]
        
        invalid_reflect = ((B4_val < 0) | (B4_val > 1) |
                          (B5_val < 0) | (B5_val > 1) |
                          (B6_val < 0) | (B6_val > 1) |
                          (B7_val < 0) | (B7_val > 1) |
                          np.isnan(B4_val) | np.isnan(B5_val) |
                          np.isnan(B6_val) | np.isnan(B7_val))
        
        H, W = B4.shape
        S2REP = np.full((H, W), np.nan)
        REP_QA = np.zeros((H, W))
        REP_QA[invalid_reflect] = 3
        valid_pixel = ~invalid_reflect
        
        numerator = ((B4_val + B7_val) / 2) - B5_val
        denominator = (B6_val - B5_val) + 1e-8
        
        zero_denominator = valid_pixel & (np.abs(denominator) < 1e-6)
        REP_QA[zero_denominator] = 2
        valid_pixel[zero_denominator] = False
        
        S2REP[valid_pixel] = 705 + 35 * (numerator[valid_pixel] / denominator[valid_pixel])
        
        rep_out_range = valid_pixel & ((S2REP < 680) | (S2REP > 760))
        REP_QA[rep_out_range] = 4
        S2REP[rep_out_range] = np.nan
        REP_QA[valid_pixel & ~rep_out_range] = 1
        
        return S2REP, REP_QA
    
    @staticmethod
    def computeIntrinsicAbsorption(ast, mineral_type):
        """Compute intrinsic absorption feature"""
        eps_val = 1e-6
        H, W, _ = ast.shape
        F_abs = np.full((H, W), np.nan, dtype=np.float32)
        
        if mineral_type.lower() == 'gold':
            cont = (ast[:, :, 2] + ast[:, :, 4]) / 2  # B3N and B5 (0-indexed: 2 and 4)
            target = ast[:, :, 2]  # B3N
            F_abs = (cont - target) / (cont + eps_val)
            F_abs = F_abs + 0.5 * (ast[:, :, 5] / (ast[:, :, 4] + eps_val))  # B6/B5
        else:
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2  # B5 and B7 (0-indexed: 4 and 6)
            target = ast[:, :, 5]  # B6
            F_abs = (cont - target) / (cont + eps_val)
        
        F_abs[np.isinf(F_abs)] = np.nan
        
        return F_abs
    
    @staticmethod
    def computeDEMIndices(dem, mineral_type, H, W, inROI):
        """Compute DEM-derived indices"""
        indices = {
            'slope': np.full((H, W), np.nan, dtype=np.float32),
            'neg_curvature': np.full((H, W), np.nan, dtype=np.float32)
        }
        
        if mineral_type.lower() != 'cave':
            return indices
        
        # Compute slope
        dy, dx = np.gradient(dem)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
        indices['slope'] = GeoUtils.mat2gray_roi(slope, inROI)
        
        # Compute negative curvature
        dxx, _ = np.gradient(dx)
        _, dyy = np.gradient(dy)
        curvature = -(dxx + dyy)
        neg_curv = np.maximum(-curvature, 0)
        indices['neg_curvature'] = GeoUtils.mat2gray_roi(neg_curv, inROI)
        
        return indices
    
    @staticmethod
    def calc_local_sum_with_nan(Z):
        """Calculate local sum with NaN handling"""
        from scipy.ndimage import convolve
        
        rows, cols = Z.shape
        local_sum = np.full((rows, cols), np.nan)
        
        # Create 3x3 kernel without center
        w = np.ones((3, 3))
        w[1, 1] = 0
        
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(Z[i, j]):
                    # Extract neighborhood
                    i_start, i_end = max(0, i-1), min(rows, i+2)
                    j_start, j_end = max(0, j-1), min(cols, j+2)
                    
                    neigh = Z[i_start:i_end, j_start:j_end]
                    mask = ~np.isnan(neigh)
                    
                    if np.any(mask):
                        # Adjust kernel to match neighborhood size
                        w_start_i = 1 - (i - i_start)
                        w_end_i = w_start_i + neigh.shape[0]
                        w_start_j = 1 - (j - j_start)
                        w_end_j = w_start_j + neigh.shape[1]
                        
                        w_local = w[w_start_i:w_end_i, w_start_j:w_end_j]
                        w_mask = w_local * mask
                        w_sum = np.sum(w_mask)
                        
                        if w_sum > 0:
                            w_mask = w_mask / w_sum
                            local_sum[i, j] = np.nansum(neigh * w_mask)
                        else:
                            local_sum[i, j] = 0
                    else:
                        local_sum[i, j] = 0
        
        local_sum[np.isinf(local_sum)] = np.nan
        
        return local_sum
    
    @staticmethod
    def getMineralThresholds(mineral_type):
        """Get mineral-specific thresholds"""
        F_thr = 0.018
        delta_thr = -2
        Moran_thr = 0.20
        
        # Enhancement function
        def enh_func(Ferric, Fe_anom, OH_anom, Clay, NDVI, *args):
            return (0.45 * Ferric + 0.25 * Fe_anom + 0.15 * OH_anom + 
                   0.10 * Clay + 0.05 * NDVI)
        
        return F_thr, delta_thr, Moran_thr, enh_func
    
    @staticmethod
    def getYakymchukParams(mineral_type):
        """Get Yakymchuk model parameters"""
        params = {'a': 50, 'b': 150, 'c': 20}
        return params
