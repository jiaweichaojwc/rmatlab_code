"""
Geographic Utilities for remote sensing data processing.
地理数据处理工具集
"""

import numpy as np
import os
import glob
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
from scipy.ndimage import convolve
from scipy.interpolate import interp2d
import warnings

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("rasterio not available. Some functions may not work.")

try:
    from shapely.geometry import Polygon, Point
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    warnings.warn("shapely not available. ROI functions may not work.")


class GeoUtils:
    """Utility class for geographic data processing."""
    
    @staticmethod
    def get_region_config(region_type: str = ''):
        """
        Get region configuration interactively or from predefined settings.
        
        Parameters:
        -----------
        region_type : str
            Region type identifier (empty for interactive mode)
            
        Returns:
        --------
        tuple
            (data_dir, roi_file, belt_lon, belt_lat)
        """
        # Note: Interactive file selection not available in Python without GUI
        # Users need to provide paths directly
        print(">>> [Python版] 请在代码中直接指定 data_dir 和 roi_file 路径")
        raise NotImplementedError("Interactive mode requires GUI - please specify paths directly in code")
    
    @staticmethod
    def read_roi_robust(roi_file: str) -> Tuple[Optional[any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Robustly read ROI coordinates from Excel/CSV file.
        
        Parameters:
        -----------
        roi_file : str
            Path to the ROI coordinate file
            
        Returns:
        --------
        tuple
            (roi_poly, inROI_vec, lonGrid, latGrid, lonROI, latROI)
        """
        try:
            # Read the file
            if roi_file.endswith('.csv'):
                data = pd.read_csv(roi_file)
            else:  # Excel file
                data = pd.read_excel(roi_file)
            
            # Try to identify lon/lat columns
            lon_col = None
            lat_col = None
            
            for col in data.columns:
                col_data = pd.to_numeric(data[col], errors='coerce')
                mean_val = col_data.mean()
                if 60 < mean_val < 160:
                    lon_col = col
                elif 0 < mean_val < 60:
                    lat_col = col
            
            if lon_col is None or lat_col is None:
                # Default to first two numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    lon_col, lat_col = numeric_cols[0], numeric_cols[1]
                else:
                    raise ValueError("无法识别经纬度列")
            
            lonROI = data[lon_col].dropna().values
            latROI = data[lat_col].dropna().values
            
            # Close the polygon if needed
            if abs(lonROI[0] - lonROI[-1]) > 1e-6 or abs(latROI[0] - latROI[-1]) > 1e-6:
                lonROI = np.append(lonROI, lonROI[0])
                latROI = np.append(latROI, latROI[0])
            
            if HAS_SHAPELY:
                roi_poly = Polygon(zip(lonROI, latROI))
            else:
                roi_poly = None
            
            return roi_poly, np.array([]), np.array([]), np.array([]), lonROI, latROI
            
        except Exception as e:
            print(f"读取坐标文件出错: {e}")
            raise
    
    @staticmethod
    def read_sentinel2(data_dir: str):
        """
        Read Sentinel-2 data from directory.
        
        Parameters:
        -----------
        data_dir : str
            Path to data directory
            
        Returns:
        --------
        tuple
            (s2_data, spatial_ref, ref_tif_path)
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required for reading Sentinel-2 data")
        
        # Find Sentinel-2 directory
        s2_dirs = glob.glob(os.path.join(data_dir, 'Sentinel*2*L2*'))
        if not s2_dirs:
            s2_dirs = glob.glob(os.path.join(data_dir, 'Sentinel*2*'))
        
        if not s2_dirs:
            raise FileNotFoundError("未找到Sentinel-2数据")
        
        s2_dir = s2_dirs[0]
        
        # Read bands
        band_patterns = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B05', 'B06', 'B07']
        s2_data = []
        ref_tif_path = None
        spatial_ref = None
        
        for pattern in band_patterns:
            files = glob.glob(os.path.join(s2_dir, f'*{pattern}*.tif')) + \
                   glob.glob(os.path.join(s2_dir, f'*{pattern}*.jp2'))
            
            if files:
                if ref_tif_path is None:
                    ref_tif_path = files[0]
                    
                with rasterio.open(files[0]) as src:
                    data = src.read(1).astype(np.float32)
                    if spatial_ref is None:
                        spatial_ref = src.profile
                    s2_data.append(data)
        
        if s2_data:
            s2_cube = np.stack(s2_data, axis=-1) * 0.0001  # Scale factor
            return s2_cube, spatial_ref, ref_tif_path
        else:
            raise FileNotFoundError("无法读取Sentinel-2波段数据")
    
    @staticmethod
    def read_landsat8(data_dir: str, spatial_ref):
        """Read Landsat 8 data."""
        # Simplified version - similar to read_sentinel2
        print("  [注意] Landsat8 读取功能简化版本")
        return None
    
    @staticmethod
    def read_aster(data_dir: str, spatial_ref):
        """Read ASTER data."""
        # Simplified version
        print("  [注意] ASTER 读取功能简化版本")
        return None
    
    @staticmethod
    def read_dem_and_roi(data_dir: str, roi_file: str, spatial_ref):
        """
        Read DEM and ROI data.
        
        Returns:
        --------
        tuple
            (dem, inROI, lonGrid, latGrid, lonROI, latROI)
        """
        # Read DEM
        dem_files = glob.glob(os.path.join(data_dir, 'DEM.tif*'))
        
        dem = None
        if dem_files and HAS_RASTERIO:
            with rasterio.open(dem_files[0]) as src:
                dem = src.read(1).astype(np.float32)
                dem[np.isinf(dem)] = np.nan
        
        # Read ROI
        roi_poly, _, _, _, lonROI, latROI = GeoUtils.read_roi_robust(roi_file)
        
        # Create coordinate grids (simplified)
        if spatial_ref and 'transform' in spatial_ref:
            height = spatial_ref['height']
            width = spatial_ref['width']
        else:
            height, width = 100, 100  # Default
        
        lonGrid = np.zeros((height, width))
        latGrid = np.zeros((height, width))
        inROI = np.ones((height, width), dtype=bool)
        
        return dem, inROI, lonGrid, latGrid, lonROI, latROI
    
    @staticmethod
    def get_band(*cubes, idx: int) -> np.ndarray:
        """
        Get a specific band from available data cubes.
        
        Parameters:
        -----------
        *cubes : array-like
            Data cubes to search
        idx : int
            Band index to retrieve
            
        Returns:
        --------
        numpy.ndarray
            Requested band data
        """
        for cube in cubes[:-1]:  # Last argument is idx
            if cube is not None and cube.shape[-1] >= idx:
                band = cube[:, :, idx-1] if cube.ndim == 3 else cube
                if np.count_nonzero(~np.isnan(band)) > 100:
                    return band
        
        # Return NaN array if not found
        if cubes[0] is not None:
            return np.full(cubes[0].shape[:2], np.nan, dtype=np.float32)
        return np.array([])
    
    @staticmethod
    def mat2gray_roi(img: np.ndarray, inROI: np.ndarray, 
                     min_val: Optional[float] = None, 
                     max_val: Optional[float] = None) -> np.ndarray:
        """
        Normalize image to [0, 1] range using ROI statistics.
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input image
        inROI : numpy.ndarray
            ROI mask
        min_val : float, optional
            Minimum value for normalization
        max_val : float, optional
            Maximum value for normalization
            
        Returns:
        --------
        numpy.ndarray
            Normalized image
        """
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
            val = np.clip(val, 0, 1)
            img_norm[inROI] = val
        
        return img_norm
    
    @staticmethod
    def calculate_s2rep_from_dn(B4, B5, B6, B7, scale_factors, offsets):
        """
        Calculate Sentinel-2 Red Edge Position (S2REP).
        
        Parameters:
        -----------
        B4, B5, B6, B7 : numpy.ndarray
            Band data
        scale_factors : list
            Scale factors for each band
        offsets : list
            Offset values for each band
            
        Returns:
        --------
        tuple
            (S2REP, REP_QA) - Red edge position and quality flag
        """
        # Convert to reflectance
        B4_val = (B4 * 10000) * scale_factors[0] + offsets[0]
        B5_val = (B5 * 10000) * scale_factors[1] + offsets[1]
        B6_val = (B6 * 10000) * scale_factors[2] + offsets[2]
        B7_val = (B7 * 10000) * scale_factors[3] + offsets[3]
        
        # Check for invalid reflectance
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
        
        # Calculate S2REP
        numerator = ((B4_val + B7_val) / 2) - B5_val
        denominator = (B6_val - B5_val) + 1e-8
        
        # Check for zero denominator
        zero_denom = valid_pixel & (np.abs(denominator) < 1e-6)
        REP_QA[zero_denom] = 2
        valid_pixel[zero_denom] = False
        
        # Calculate red edge position
        S2REP[valid_pixel] = 705 + 35 * (numerator[valid_pixel] / denominator[valid_pixel])
        
        # Check range
        rep_out_range = valid_pixel & ((S2REP < 680) | (S2REP > 760))
        REP_QA[rep_out_range] = 4
        S2REP[rep_out_range] = np.nan
        REP_QA[valid_pixel & ~rep_out_range] = 1
        
        return S2REP, REP_QA
    
    @staticmethod
    def compute_intrinsic_absorption(ast: np.ndarray, mineral_type: str) -> np.ndarray:
        """
        Compute intrinsic absorption from ASTER data.
        
        Parameters:
        -----------
        ast : numpy.ndarray
            ASTER data cube
        mineral_type : str
            Type of mineral
            
        Returns:
        --------
        numpy.ndarray
            Absorption intensity map
        """
        eps_val = 1e-6
        H, W, _ = ast.shape
        F_abs = np.full((H, W), np.nan, dtype=np.float32)
        
        if mineral_type.lower() == 'gold':
            cont = (ast[:, :, 2] + ast[:, :, 4]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.5 * (ast[:, :, 5] / (ast[:, :, 4] + eps_val))
        else:
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2
            target = ast[:, :, 5]
            F_abs = (cont - target) / (cont + eps_val)
        
        F_abs[np.isinf(F_abs)] = np.nan
        return F_abs
    
    @staticmethod
    def calc_local_sum_with_nan(Z: np.ndarray) -> np.ndarray:
        """
        Calculate local sum with NaN handling (for Moran I calculation).
        
        Parameters:
        -----------
        Z : numpy.ndarray
            Input z-score normalized array
            
        Returns:
        --------
        numpy.ndarray
            Local sum values
        """
        rows, cols = Z.shape
        local_sum = np.full_like(Z, np.nan)
        
        # Kernel (3x3 with center = 0)
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        
        # Pad with NaN
        Z_padded = np.pad(Z, 1, mode='constant', constant_values=np.nan)
        
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(Z[i, j]):
                    neigh = Z_padded[i:i+3, j:j+3]
                    mask = ~np.isnan(neigh)
                    
                    if np.any(mask):
                        w_mask = kernel * mask
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
    def get_mineral_thresholds(mineral_type: str) -> Tuple[float, float, float, any]:
        """
        Get thresholds for mineral detection.
        
        Parameters:
        -----------
        mineral_type : str
            Type of mineral
            
        Returns:
        --------
        tuple
            (F_thr, delta_thr, Moran_thr, enh_func)
        """
        F_thr = 0.018
        delta_thr = -2
        Moran_thr = 0.20
        
        def enh_func(Ferric, Fe_anom, OH_anom, Clay, NDVI):
            return 0.45*Ferric + 0.25*Fe_anom + 0.15*OH_anom + 0.10*Clay + 0.05*NDVI
        
        return F_thr, delta_thr, Moran_thr, enh_func
