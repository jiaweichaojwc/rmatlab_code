"""
GeoUtils: Geospatial utility functions for mineral detection from remote sensing data.

This module provides static methods for:
- Region configuration and coordinate handling
- ROI (Region of Interest) reading from Excel/CSV files
- Multi-sensor data reading (Sentinel-2, Landsat-8, ASTER)
- DEM (Digital Elevation Model) processing
- Spectral indices calculation
- Mineral-specific absorption feature computation
- Spatial analysis and enhancement functions
"""

import os
import warnings
from typing import Tuple, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon, Point
import glob


class GeoUtils:
    """Static utility class for geospatial operations in mineral detection."""

    # ================= 1. Region Configuration and Path Retrieval =================
    
    @staticmethod
    def get_region_config(region_type: str) -> Tuple[str, str, np.ndarray, np.ndarray]:
        """
        Interactive mode: manually select data folder and coordinate file.
        
        Note: This method uses user interaction (file dialogs) which is not available
        in Python CLI. In practice, you should modify this to accept paths as parameters.
        
        Args:
            region_type: Type of region (for future use)
            
        Returns:
            Tuple of (data_dir, roi_file, belt_lon, belt_lat)
        """
        raise NotImplementedError(
            "Interactive file selection not supported in Python CLI. "
            "Please use direct paths or implement custom selection logic."
        )
    
    @staticmethod
    def get_belt_coords(root_dir: str, roi_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding coordinates for the mineral belt.
        
        Args:
            root_dir: Root directory path
            roi_file: ROI file path (can be relative or absolute)
            
        Returns:
            Tuple of (belt_lon, belt_lat) as closed rectangular arrays
        """
        # Determine full path
        if os.path.isfile(roi_file):
            fullpath_roi = roi_file
        else:
            fullpath_roi = os.path.join(root_dir, roi_file)
        
        # Get valid coordinates using robust reading
        _, _, _, _, lon_valid, lat_valid = GeoUtils.read_roi_robust(fullpath_roi)
        
        # Create closed rectangular bounding box
        x1 = round(np.min(lon_valid), 4)
        x2 = round(np.max(lon_valid), 4)
        y1 = round(np.min(lat_valid), 4)
        y2 = round(np.max(lat_valid), 4)
        
        belt_lon = np.array([x1, x2, x2, x1, x1])
        belt_lat = np.array([y1, y1, y2, y2, y1])
        
        print(f"    Auto-detected coordinate range: Lon[{x1:.2f} ~ {x2:.2f}], Lat[{y1:.2f} ~ {y2:.2f}]")
        
        return belt_lon, belt_lat
    
    # ================= 2. Intelligent ROI Reading Function =================
    
    @staticmethod
    def read_roi_robust(fullpath_roi: str) -> Tuple[
        Polygon, Optional[np.ndarray], Optional[np.ndarray], 
        Optional[np.ndarray], np.ndarray, np.ndarray
    ]:
        """
        Robustly read ROI coordinates from Excel/CSV file.
        
        Intelligently detects longitude and latitude columns based on value ranges.
        
        Args:
            fullpath_roi: Full path to ROI coordinate file
            
        Returns:
            Tuple of (roi_poly, in_roi_vec, lon_grid, lat_grid, lon_roi, lat_roi)
        """
        try:
            # Read file based on extension
            file_ext = os.path.splitext(fullpath_roi)[1].lower()
            if file_ext in ['.xlsx', '.xls']:
                raw_data = pd.read_excel(fullpath_roi, header=None)
            elif file_ext == '.csv':
                raw_data = pd.read_csv(fullpath_roi, header=None)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            rows, cols = raw_data.shape
            
            # Intelligently detect longitude and latitude columns
            candidates_lon = []
            candidates_lat = []
            parsed_cols = []
            
            for c in range(cols):
                col_data = raw_data.iloc[:, c]
                nums = np.full(rows, np.nan)
                valid_count = 0
                
                for r in range(rows):
                    val = col_data.iloc[r]
                    if pd.notna(val):
                        try:
                            nums[r] = float(val)
                            valid_count += 1
                        except (ValueError, TypeError):
                            nums[r] = np.nan
                    else:
                        nums[r] = np.nan
                
                if valid_count > 3:
                    valid_nums = nums[~np.isnan(nums)]
                    mean_val = np.mean(valid_nums)
                    min_val = np.min(valid_nums)
                    max_val = np.max(valid_nums)
                    parsed_cols.append(nums)
                    
                    # Feature-based detection
                    if 60 < mean_val < 160 and (max_val - min_val) < 20:
                        candidates_lon.append(c)
                    if 0 < mean_val < 60 and (max_val - min_val) < 20:
                        candidates_lat.append(c)
                else:
                    parsed_cols.append(nums)
            
            # Decision logic
            if len(candidates_lon) == 1 and len(candidates_lat) == 1:
                final_lon_col = candidates_lon[0]
                final_lat_col = candidates_lat[0]
            else:
                if cols >= 3 and np.any(~np.isnan(parsed_cols[1])):
                    final_lon_col = 1
                    final_lat_col = 2
                elif cols >= 2 and np.any(~np.isnan(parsed_cols[0])):
                    final_lon_col = 0
                    final_lat_col = 1
                else:
                    raise ValueError("Cannot identify longitude/latitude columns. Please check file format.")
            
            lon_roi = parsed_cols[final_lon_col]
            lat_roi = parsed_cols[final_lat_col]
            
            # Filter valid coordinates
            valid_mask = ~np.isnan(lon_roi) & ~np.isnan(lat_roi)
            lon_roi = lon_roi[valid_mask]
            lat_roi = lat_roi[valid_mask]
            
            if len(lon_roi) == 0:
                raise ValueError("No valid coordinate data found")
            
            # Close the polygon if not already closed
            if abs(lon_roi[0] - lon_roi[-1]) > 1e-6 or abs(lat_roi[0] - lat_roi[-1]) > 1e-6:
                lon_roi = np.append(lon_roi, lon_roi[0])
                lat_roi = np.append(lat_roi, lat_roi[0])
            
            # Create polygon
            roi_poly = Polygon(zip(lon_roi, lat_roi))
            in_roi_vec = None
            lon_grid = None
            lat_grid = None
            
            return roi_poly, in_roi_vec, lon_grid, lat_grid, lon_roi, lat_roi
            
        except Exception as e:
            print(f"Error reading coordinate file: {str(e)}")
            raise
    
    # ================= 3. Data Reading Wrappers =================
    
    @staticmethod
    def read_sentinel2(data_dir: str) -> Tuple[np.ndarray, Any, str]:
        """
        Read Sentinel-2 L2A data.
        
        Args:
            data_dir: Directory containing Sentinel-2 data
            
        Returns:
            Tuple of (s2_cube, raster_reference, reference_tif_path)
            s2_cube shape: (H, W, 9) - 9 bands
        """
        # Find Sentinel-2 directory
        s2_pattern = os.path.join(data_dir, 'Sentinel*2 L2*')
        s2_dirs = glob.glob(s2_pattern)
        
        if not s2_dirs:
            raise FileNotFoundError("Sentinel-2 L2A data not found")
        
        s2_dir = s2_dirs[0]
        
        # Find reference file (B08 preferred)
        files = glob.glob(os.path.join(s2_dir, '*B08*'))
        if not files:
            files = glob.glob(os.path.join(s2_dir, '*.tif*'))
        if not files:
            files = glob.glob(os.path.join(s2_dir, '*.jp2'))
        
        if not files:
            raise FileNotFoundError("No Sentinel-2 files found")
        
        ref_tif_path = files[0]
        
        # Read reference for georeferencing
        with rasterio.open(ref_tif_path) as src:
            R = {
                'height': src.height,
                'width': src.width,
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds
            }
        
        # Band patterns
        s2_patterns = [['B02'], ['B03'], ['B04'], ['B08'], ['B11'], ['B12'], 
                      ['B05'], ['B06'], ['B07']]
        
        s2_raw = GeoUtils._read_multi_bands_smart(s2_dir, s2_patterns, R, 9)
        s2 = s2_raw.astype(np.float32) * 0.0001
        
        return s2, R, ref_tif_path
    
    @staticmethod
    def read_landsat8(data_dir: str, R: Dict[str, Any]) -> np.ndarray:
        """
        Read Landsat-8 data.
        
        Args:
            data_dir: Directory containing Landsat data
            R: Reference raster information dictionary
            
        Returns:
            Landsat cube of shape (H, W, 7)
        """
        # Find Landsat directory
        lan_l1 = glob.glob(os.path.join(data_dir, '*Landsat*8*L1*'))
        lan_l2 = glob.glob(os.path.join(data_dir, '*Landsat*8*L2*'))
        
        if lan_l1:
            lan_dir = lan_l1[0]
        elif lan_l2:
            lan_dir = lan_l2[0]
        else:
            raise FileNotFoundError("Landsat 8 data not found")
        
        lan_patterns = [['B2'], ['B3'], ['B4'], ['B5'], ['B6'], ['B7'], ['B8']]
        lan = GeoUtils._read_multi_bands_smart(lan_dir, lan_patterns, R, 7)
        
        return lan
    
    @staticmethod
    def read_aster(data_dir: str, R: Dict[str, Any]) -> np.ndarray:
        """
        Read ASTER data with thermal band calibration.
        
        Args:
            data_dir: Directory containing ASTER data
            R: Reference raster information dictionary
            
        Returns:
            ASTER cube of shape (H, W, 14)
        """
        # Find ASTER directory
        aster_dirs = glob.glob(os.path.join(data_dir, '*ASTER*L2*'))
        if not aster_dirs:
            aster_dirs = glob.glob(os.path.join(data_dir, '*ASTER*L1*'))
        
        if not aster_dirs:
            raise FileNotFoundError("ASTER data not found")
        
        aster_dir = aster_dirs[0]
        
        # Band patterns
        aster_pat = [
            ['B01', 'B1'], ['B02', 'B2'], ['B3N', 'B03N'],
            ['B04', 'B4'], ['B05', 'B5'], ['B06', 'B6'],
            ['B07', 'B7'], ['B08', 'B8'], ['B09'], ['B10'],
            ['B11'], ['B12'], ['B13'], ['B14']
        ]
        
        H, W = R['height'], R['width']
        ast = np.full((H, W, 14), np.nan, dtype=np.float32)
        
        for b in range(14):
            single_band = GeoUtils._read_any_smart(aster_dir, aster_pat[b], R)
            
            if b < 9:
                # VNIR and SWIR bands: apply scale factor
                single_band = single_band * 0.01
                single_band[np.isinf(single_band)] = np.nan
            else:
                # Thermal bands: convert to temperature
                single_band = single_band * 0.1 + 300
                single_band[np.isinf(single_band)] = 300
            
            ast[:, :, b] = single_band
        
        return ast
    
    @staticmethod
    def read_dem_and_roi(
        data_dir: str, 
        roi_file: str, 
        R: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read DEM and ROI, create coordinate grids.
        
        Args:
            data_dir: Directory containing DEM file
            roi_file: Path to ROI coordinate file
            R: Reference raster information dictionary
            
        Returns:
            Tuple of (dem, in_roi, lon_grid, lat_grid, lon_roi, lat_roi)
        """
        # Find DEM file
        dem_files = glob.glob(os.path.join(data_dir, 'DEM.tif'))
        if not dem_files:
            dem_files = glob.glob(os.path.join(data_dir, 'DEM.tiff'))
        
        H, W = R['height'], R['width']
        bounds = R['bounds']
        
        # Create coordinate grids
        lon_vec = np.linspace(bounds.left, bounds.right, W)
        lat_vec = np.linspace(bounds.top, bounds.bottom, H)
        lon_grid, lat_grid = np.meshgrid(lon_vec, lat_vec)
        
        # Determine full ROI path
        if os.path.isfile(roi_file):
            fullpath = roi_file
        else:
            root_dir = os.path.dirname(data_dir)
            fullpath = os.path.join(root_dir, roi_file)
        
        # Read ROI
        roi_poly, _, _, _, lon_roi, lat_roi = GeoUtils.read_roi_robust(fullpath)
        
        # Create mask for points inside ROI
        points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        in_roi_vec = np.array([roi_poly.contains(Point(p)) for p in points])
        in_roi = np.flipud(in_roi_vec.reshape(H, W))
        
        # Read DEM if available
        if dem_files:
            with rasterio.open(dem_files[0]) as src:
                dem_raw = src.read(1)
            
            dem = dem_raw.astype(np.float32)
            dem[np.isinf(dem)] = np.nan
            
            # Resample if size mismatch
            if dem.shape != (H, W):
                dem = GeoUtils._resample_to_reference(dem, R)
        else:
            dem = np.full((H, W), np.nan, dtype=np.float32)
        
        return dem, in_roi, lon_grid, lat_grid, lon_roi, lat_roi
    
    # ================= 4. Auxiliary Functions =================
    
    @staticmethod
    def _read_multi_bands_smart(
        dir_path: str, 
        patterns_cell: list, 
        R: Dict[str, Any], 
        num_bands: int
    ) -> np.ndarray:
        """
        Read multiple bands with smart pattern matching and resampling.
        
        Args:
            dir_path: Directory containing band files
            patterns_cell: List of lists containing pattern strings for each band
            R: Reference raster information
            num_bands: Number of bands to read
            
        Returns:
            Cube of shape (H, W, num_bands)
        """
        H, W = R['height'], R['width']
        cube = np.full((H, W, num_bands), np.nan, dtype=np.float32)
        
        # Find all potential files
        files = []
        for ext in ['*.tif', '*.tiff', '*.jp2']:
            files.extend(glob.glob(os.path.join(dir_path, ext)))
        
        for b in range(num_bands):
            pattern = patterns_cell[b]
            
            for file_path in files:
                fname_upper = os.path.basename(file_path).upper()
                
                # Check if any pattern matches
                if any(p.upper() in fname_upper for p in pattern):
                    with rasterio.open(file_path) as src:
                        A = src.read(1).astype(np.float32)
                    
                    # Resample if needed
                    if A.shape != (H, W):
                        A = GeoUtils._resample_to_reference(A, R)
                    
                    cube[:, :, b] = A
                    break
        
        return cube
    
    @staticmethod
    def _read_any_smart(
        dir_path: str, 
        patterns_cell: list, 
        R: Dict[str, Any]
    ) -> np.ndarray:
        """
        Read a single band with smart pattern matching.
        
        Args:
            dir_path: Directory containing band file
            patterns_cell: List of pattern strings
            R: Reference raster information
            
        Returns:
            2D array of shape (H, W)
        """
        if isinstance(patterns_cell, str):
            patterns_cell = [patterns_cell]
        
        cube = GeoUtils._read_multi_bands_smart(dir_path, [patterns_cell], R, 1)
        return cube[:, :, 0]
    
    @staticmethod
    def _resample_to_reference(
        array: np.ndarray, 
        R: Dict[str, Any]
    ) -> np.ndarray:
        """
        Resample array to match reference dimensions using bilinear interpolation.
        
        Args:
            array: Input 2D array
            R: Reference raster information
            
        Returns:
            Resampled array of shape (R['height'], R['width'])
        """
        H_src, W_src = array.shape
        H_dst, W_dst = R['height'], R['width']
        bounds = R['bounds']
        
        # Create coordinate arrays for source
        lon_src = np.linspace(bounds.left, bounds.right, W_src)
        lat_src = np.linspace(bounds.top, bounds.bottom, H_src)
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (lat_src, lon_src), 
            array, 
            method='linear', 
            bounds_error=False, 
            fill_value=np.nan
        )
        
        # Create target coordinates
        lon_dst = np.linspace(bounds.left, bounds.right, W_dst)
        lat_dst = np.linspace(bounds.top, bounds.bottom, H_dst)
        lon_grid, lat_grid = np.meshgrid(lon_dst, lat_dst)
        
        # Interpolate
        points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        result = interpolator(points).reshape(H_dst, W_dst)
        
        return result.astype(np.float32)
    
    @staticmethod
    def get_band(*cubes, idx: int) -> np.ndarray:
        """
        Get band from multiple possible cube sources.
        
        Tries to extract band idx from each cube in order, returning the first
        valid band (with sufficient non-zero, non-NaN values).
        
        Args:
            *cubes: Variable number of data cubes
            idx: Band index (1-based, converted to 0-based internally)
            
        Returns:
            2D array of the requested band
        """
        idx_0 = idx - 1  # Convert to 0-based indexing
        
        for cube in cubes:
            if cube.ndim == 3 and cube.shape[2] > idx_0:
                band = cube[:, :, idx_0]
                valid_count = np.sum((band != 0) & ~np.isnan(band))
                if valid_count > 100:
                    return band
        
        # Return NaN array if no valid band found
        H, W = cubes[0].shape[:2]
        return np.full((H, W), np.nan, dtype=np.float32)
    
    @staticmethod
    def mat2gray_roi(
        img: np.ndarray, 
        in_roi: np.ndarray, 
        min_val: Optional[float] = None, 
        max_val: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize image to [0, 1] range based on ROI statistics.
        
        Args:
            img: Input image
            in_roi: Boolean mask for ROI
            min_val: Optional minimum value (computed from ROI if not provided)
            max_val: Optional maximum value (computed from ROI if not provided)
            
        Returns:
            Normalized image
        """
        img_norm = np.full(img.shape, np.nan, dtype=np.float32)
        
        # Get valid ROI values
        img_roi = img[in_roi]
        img_roi = img_roi[~np.isnan(img_roi) & ~np.isinf(img_roi)]
        
        if len(img_roi) == 0:
            return img_norm
        
        # Determine min/max
        if min_val is None:
            min_val = np.min(img_roi)
        if max_val is None:
            max_val = np.max(img_roi)
        
        # Normalize
        if max_val - min_val < np.finfo(float).eps:
            img_norm[in_roi] = 0.5
        else:
            val = (img[in_roi] - min_val) / (max_val - min_val)
            val = np.clip(val, 0, 1)
            img_norm[in_roi] = val
        
        return img_norm
    
    @staticmethod
    def calculate_s2rep_from_dn(
        B4: np.ndarray, 
        B5: np.ndarray, 
        B6: np.ndarray, 
        B7: np.ndarray,
        scale_factors: Tuple[float, float, float, float],
        offsets: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Sentinel-2 Red Edge Position (S2REP) from Digital Numbers.
        
        Args:
            B4, B5, B6, B7: Band arrays (665, 705, 740, 783 nm)
            scale_factors: Scale factors for each band
            offsets: Offset values for each band
            
        Returns:
            Tuple of (S2REP, REP_QA) where:
                S2REP: Red Edge Position in nm
                REP_QA: Quality flags (0: invalid, 1: valid, 2: zero denominator, 
                        3: invalid reflectance, 4: out of range)
        """
        # Convert DN to reflectance
        B4_val = (B4 * 10000).astype(np.float64) * scale_factors[0] + offsets[0]
        B5_val = (B5 * 10000).astype(np.float64) * scale_factors[1] + offsets[1]
        B6_val = (B6 * 10000).astype(np.float64) * scale_factors[2] + offsets[2]
        B7_val = (B7 * 10000).astype(np.float64) * scale_factors[3] + offsets[3]
        
        # Check for invalid reflectance
        invalid_reflect = (
            (B4_val < 0) | (B4_val > 1) |
            (B5_val < 0) | (B5_val > 1) |
            (B6_val < 0) | (B6_val > 1) |
            (B7_val < 0) | (B7_val > 1) |
            np.isnan(B4_val) | np.isnan(B5_val) | np.isnan(B6_val) | np.isnan(B7_val)
        )
        
        H, W = B4.shape
        S2REP = np.full((H, W), np.nan, dtype=np.float32)
        REP_QA = np.zeros((H, W), dtype=np.int32)
        
        # Mark invalid reflectance
        REP_QA[invalid_reflect] = 3
        valid_pixel = ~invalid_reflect
        
        # Calculate S2REP
        numerator = ((B4_val + B7_val) / 2) - B5_val
        denominator = (B6_val - B5_val) + 1e-8
        
        # Mark zero denominator
        zero_denominator = valid_pixel & (np.abs(denominator) < 1e-6)
        REP_QA[zero_denominator] = 2
        valid_pixel = valid_pixel & ~zero_denominator
        
        # Compute REP
        S2REP[valid_pixel] = 705 + 35 * (numerator[valid_pixel] / denominator[valid_pixel])
        
        # Mark out of range
        rep_out_range = valid_pixel & ((S2REP < 680) | (S2REP > 760))
        REP_QA[rep_out_range] = 4
        S2REP[rep_out_range] = np.nan
        
        # Mark valid
        REP_QA[valid_pixel & ~rep_out_range] = 1
        
        return S2REP, REP_QA
    
    # ================= 5. Intrinsic Absorption Spectrum Logic =================
    
    @staticmethod
    def compute_intrinsic_absorption(ast: np.ndarray, mineral_type: str) -> np.ndarray:
        """
        Compute intrinsic absorption features for specific mineral types.
        
        ASTER bands: 1(0.56μm), 2(0.66μm), 3N(0.81μm), 4(1.6μm), 5(2.1μm), 
                    6(2.2μm), 7(2.3μm), 8(2.5μm), 9-14(thermal)
        
        Args:
            ast: ASTER data cube of shape (H, W, 14)
            mineral_type: Type of mineral (e.g., 'gold', 'copper', 'iron')
            
        Returns:
            Absorption feature map of shape (H, W)
        """
        eps_val = 1e-6
        H, W = ast.shape[:2]
        F_abs = np.full((H, W), np.nan, dtype=np.float32)
        
        mineral_type_lower = mineral_type.lower()
        
        if mineral_type_lower == 'gold':
            # Pyrite (Fe-S: 0.8-0.9μm) + Al-OH (2.2μm)
            cont = (ast[:, :, 2] + ast[:, :, 4]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.5 * (ast[:, :, 5] / (ast[:, :, 4] + eps_val))
        
        elif mineral_type_lower == 'copper':
            # Cu²⁺ (0.8-0.9μm) + OH (2.2μm)
            cont = (ast[:, :, 2] + ast[:, :, 4]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.5 * (ast[:, :, 5] / (ast[:, :, 4] + eps_val))
        
        elif mineral_type_lower == 'iron':
            # Iron oxides (0.8μm)
            cont = (ast[:, :, 1] + ast[:, :, 3]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'coal':
            # Organic matter (2.3μm)
            cont = (ast[:, :, 4] + ast[:, :, 7]) / 2
            target = ast[:, :, 6]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'rare_earth':
            # REE electronic transition (2.2μm)
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2
            target = ast[:, :, 5]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'silver':
            # Galena (Pb-S: 1.0μm) + OH (2.2μm)
            cont = (ast[:, :, 2] + ast[:, :, 3]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.4 * (ast[:, :, 5] / (ast[:, :, 4] + eps_val))
        
        elif mineral_type_lower == 'aluminum':
            # Kaolinite (Al-OH: 2.2μm)
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2
            target = ast[:, :, 5]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.3 * (ast[:, :, 3] / (ast[:, :, 4] + eps_val))
        
        elif mineral_type_lower == 'lead':
            # Galena (Pb-S: 1.0μm)
            cont = (ast[:, :, 2] + ast[:, :, 3]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'zinc':
            # Sphalerite (Zn-Fe: 0.9-1.1μm)
            cont = (ast[:, :, 2] + ast[:, :, 3]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.2 * (ast[:, :, 5] / (ast[:, :, 6] + eps_val))
        
        elif mineral_type_lower == 'nickel':
            # Nickel silicate (Ni-OH/Mg-OH: 1.8-2.3μm)
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2
            target = ast[:, :, 5]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.3 * (ast[:, :, 3] / (ast[:, :, 4] + eps_val))
        
        elif mineral_type_lower == 'cobalt':
            # Heterogenite (Co²⁺: 0.5-0.6μm)
            cont = (ast[:, :, 0] + ast[:, :, 1]) / 2
            target = ast[:, :, 0]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'molybdenum':
            # Molybdenite (Fe-related: 0.9μm)
            cont = (ast[:, :, 1] + ast[:, :, 2]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'fluorite':
            # Fluorite (weak OH: 1.4μm, using 2.2μm as proxy)
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2
            target = ast[:, :, 5]
            F_abs = (cont - target) / (cont + eps_val) * 0.5
        
        elif mineral_type_lower == 'tin':
            # Cassiterite (Sn-Fe: 1.0μm + OH: 2.2μm)
            cont = (ast[:, :, 2] + ast[:, :, 4]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.4 * (ast[:, :, 5] / (ast[:, :, 4] + eps_val))
        
        elif mineral_type_lower == 'tungsten':
            # Wolframite (W-Fe: 0.9-1.0μm)
            cont = (ast[:, :, 1] + ast[:, :, 2]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'petroleum':
            # Petroleum (C-H: 1.7-1.75μm + 2.3-2.5μm)
            cont = (ast[:, :, 3] + ast[:, :, 7]) / 2
            target = ast[:, :, 6]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.3 * (ast[:, :, 3] / (ast[:, :, 4] + eps_val))
        
        elif mineral_type_lower == 'gas':
            # Natural gas (CH₄: 1.65-1.7μm + 2.3μm)
            cont = (ast[:, :, 3] + ast[:, :, 6]) / 2
            target = ast[:, :, 3]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'coalbed_gas':
            # Coalbed methane (C-H: 1.7μm + 2.3μm)
            cont = (ast[:, :, 4] + ast[:, :, 7]) / 2
            target = ast[:, :, 6]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.2 * (ast[:, :, 3] / (ast[:, :, 4] + eps_val))
        
        elif mineral_type_lower == 'helium':
            # Helium (no direct absorption, using host rock OH)
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2
            target = ast[:, :, 5]
            F_abs = (cont - target) / (cont + eps_val) * 0.2
        
        elif mineral_type_lower == 'lithium':
            # Spodumene (Li-OH/Al-OH: 2.2-2.4μm)
            cont = (ast[:, :, 5] + ast[:, :, 7]) / 2
            target = ast[:, :, 6]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.3 * (ast[:, :, 5] / (ast[:, :, 6] + eps_val))
        
        elif mineral_type_lower == 'natural_hydrogen':
            # Natural hydrogen (no direct absorption, using host rock Fe)
            cont = (ast[:, :, 1] + ast[:, :, 2]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val) * 0.2
        
        elif mineral_type_lower == 'potassium':
            # K-feldspar (weak OH: 1.4μm, using 2.2μm as proxy)
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2
            target = ast[:, :, 5]
            F_abs = (cont - target) / (cont + eps_val) * 0.3
        
        elif mineral_type_lower == 'uranium':
            # Uraninite (U⁴⁺/U⁶⁺: 0.8-1.0μm)
            cont = (ast[:, :, 1] + ast[:, :, 3]) / 2
            target = ast[:, :, 2]
            F_abs = (cont - target) / (cont + eps_val)
        
        elif mineral_type_lower == 'cave':
            # Cave (no mineral absorption, using DEM-derived indices)
            F_abs = np.full((H, W), np.nan, dtype=np.float32)
        
        elif mineral_type_lower == 'offshore_petroleum':
            # Offshore petroleum (same as petroleum, enhanced 2.5μm)
            cont = (ast[:, :, 3] + ast[:, :, 7]) / 2
            target = ast[:, :, 6]
            F_abs = (cont - target) / (cont + eps_val)
            F_abs += 0.4 * (ast[:, :, 7] / (ast[:, :, 6] + eps_val))
        
        else:
            # Default: generic OH absorption (2.2μm)
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2
            target = ast[:, :, 5]
            F_abs = (cont - target) / (cont + eps_val)
        
        # Remove infinities
        F_abs[np.isinf(F_abs)] = np.nan
        
        return F_abs
    
    @staticmethod
    def compute_dem_indices(
        dem: np.ndarray, 
        mineral_type: str, 
        H: int, 
        W: int, 
        in_roi: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute DEM-derived indices (slope and negative curvature).
        
        Only computed for 'cave' mineral type.
        
        Args:
            dem: Digital elevation model
            mineral_type: Type of mineral
            H: Height of output
            W: Width of output
            in_roi: Boolean ROI mask
            
        Returns:
            Dictionary with 'slope' and 'neg_curvature' arrays
        """
        indices = {
            'slope': np.full((H, W), np.nan, dtype=np.float32),
            'neg_curvature': np.full((H, W), np.nan, dtype=np.float32)
        }
        
        if mineral_type.lower() != 'cave':
            return indices
        
        # Compute gradient
        dy, dx = np.gradient(dem)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
        indices['slope'] = GeoUtils.mat2gray_roi(slope, in_roi)
        
        # Compute curvature
        dxx = np.gradient(dx, axis=1)
        dyy = np.gradient(dy, axis=0)
        curvature = -(dxx + dyy)
        neg_curv = np.maximum(-curvature, 0)
        indices['neg_curvature'] = GeoUtils.mat2gray_roi(neg_curv, in_roi)
        
        return indices
    
    @staticmethod
    def calc_local_sum_with_nan(Z: np.ndarray) -> np.ndarray:
        """
        Calculate local neighborhood sum (ignoring NaN values).
        
        Uses 3x3 window excluding center pixel.
        
        Args:
            Z: Input 2D array
            
        Returns:
            Local sum array with same shape as input
        """
        rows, cols = Z.shape
        pad = 1
        
        # Pad with NaN
        Z_padded = np.pad(Z, pad, mode='constant', constant_values=np.nan)
        local_sum = np.full((rows, cols), np.nan, dtype=np.float32)
        
        # 3x3 kernel excluding center
        w = np.ones((3, 3), dtype=np.float32)
        w[1, 1] = 0
        
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(Z[i, j]):
                    neigh = Z_padded[i:i+3, j:j+3]
                    mask = ~np.isnan(neigh)
                    
                    if np.any(mask):
                        w_mask = w * mask
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
    
    # ================= 6. Mineral Thresholds and Enhancement Formulas =================
    
    @staticmethod
    def get_mineral_thresholds(mineral_type: str) -> Tuple[float, float, float, Callable]:
        """
        Get mineral-specific thresholds and enhancement function.
        
        Args:
            mineral_type: Type of mineral
            
        Returns:
            Tuple of (F_thr, delta_thr, Moran_thr, enh_func)
        """
        mineral_type_lower = mineral_type.lower()
        
        if mineral_type_lower == 'gold':
            F_thr, delta_thr, Moran_thr = 0.018, -2, 0.20
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.45*Ferric + 0.25*Fe_anomaly + 0.15*Hydroxy_anomaly + 0.10*Clay + 0.05*NDVI_inv
        
        elif mineral_type_lower == 'copper':
            F_thr, delta_thr, Moran_thr = 0.020, -3, 0.25
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.40*Clay + 0.30*Hydroxy_anomaly + 0.20*Ferric + 0.10*Fe_anomaly
        
        elif mineral_type_lower == 'iron':
            F_thr, delta_thr, Moran_thr = 0.030, -4, 0.35
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.60*Ferric + 0.40*Fe_anomaly
        
        elif mineral_type_lower == 'lead':
            F_thr, delta_thr, Moran_thr = 0.025, -3, 0.30
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.40*Hydroxy_anomaly + 0.30*Clay + 0.30*Ferric
        
        elif mineral_type_lower == 'zinc':
            F_thr, delta_thr, Moran_thr = 0.024, -3, 0.28
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.40*Hydroxy_anomaly + 0.30*Clay + 0.30*Ferric
        
        elif mineral_type_lower == 'molybdenum':
            F_thr, delta_thr, Moran_thr = 0.028, -4, 0.32
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.50*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric
        
        elif mineral_type_lower == 'copper_gold':
            F_thr, delta_thr, Moran_thr = 0.019, -2.5, 0.22
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.40*Ferric + 0.40*Clay + 0.20*Hydroxy_anomaly
        
        elif mineral_type_lower == 'coal':
            F_thr, delta_thr, Moran_thr = 0.032, -4.5, 0.38
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.60*NDVI_inv + 0.40*Hydroxy_anomaly
        
        elif mineral_type_lower == 'tin':
            F_thr, delta_thr, Moran_thr = 0.023, -2.5, 0.26
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.50*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric
        
        elif mineral_type_lower == 'petroleum':
            F_thr, delta_thr, Moran_thr = 0.035, -5, 0.40
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.70*NDVI_inv + 0.30*Hydroxy_anomaly
        
        elif mineral_type_lower == 'gas':
            F_thr, delta_thr, Moran_thr = 0.033, -4.5, 0.38
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.60*NDVI_inv + 0.40*Hydroxy_anomaly
        
        elif mineral_type_lower == 'lithium':
            F_thr, delta_thr, Moran_thr = 0.022, -2, 0.25
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.60*Clay + 0.40*Hydroxy_anomaly
        
        elif mineral_type_lower == 'nickel':
            F_thr, delta_thr, Moran_thr = 0.026, -3.5, 0.30
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.50*Hydroxy_anomaly + 0.30*Ferric + 0.20*Clay
        
        elif mineral_type_lower == 'fluorite':
            F_thr, delta_thr, Moran_thr = 0.029, -4, 0.35
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.50*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric
        
        elif mineral_type_lower == 'phosphate':
            F_thr, delta_thr, Moran_thr = 0.027, -3.5, 0.32
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.40*Hydroxy_anomaly + 0.30*Clay + 0.30*NDVI_inv
        
        elif mineral_type_lower == 'rare_earth':
            F_thr, delta_thr, Moran_thr = 0.026, -3, 0.28
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.40*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric + 0.10*NDVI_inv
        
        elif mineral_type_lower == 'helium':
            F_thr, delta_thr, Moran_thr = 0.031, -4, 0.36
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.50*NDVI_inv + 0.30*Hydroxy_anomaly + 0.20*Clay
        
        elif mineral_type_lower == 'uranium':
            F_thr, delta_thr, Moran_thr = 0.028, -4.5, 0.32
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.40*Fe_anomaly + 0.30*Ferric + 0.20*NDVI_inv + 0.10*Hydroxy_anomaly
        
        elif mineral_type_lower == 'natural_hydrogen':
            F_thr, delta_thr, Moran_thr = 0.032, -4, 0.37
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.50*NDVI_inv + 0.30*Clay + 0.20*Hydroxy_anomaly
        
        elif mineral_type_lower == 'potassium':
            F_thr, delta_thr, Moran_thr = 0.025, -3, 0.28
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.40*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric + 0.10*NDVI_inv
        
        elif mineral_type_lower == 'cave':
            F_thr, delta_thr, Moran_thr = 0.025, -3, 0.30
            # Cave mode has additional slope and curvature parameters
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv, slope, curvature: \
                0.30*NDVI_inv + 0.25*slope + 0.20*curvature + 0.15*Hydroxy_anomaly + 0.10*Clay
        
        elif mineral_type_lower == 'offshore_petroleum':
            F_thr, delta_thr, Moran_thr = 0.030, -4, 0.35
            # Offshore petroleum mode has additional OSI and SDS parameters
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv, OSI, SDS: \
                0.40*OSI + 0.30*SDS + 0.20*NDVI_inv + 0.10*Hydroxy_anomaly
        
        else:
            # Default: gold parameters
            F_thr, delta_thr, Moran_thr = 0.018, -2, 0.20
            enh_func = lambda Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv: \
                0.45*Ferric + 0.25*Fe_anomaly + 0.15*Hydroxy_anomaly + 0.10*Clay + 0.05*NDVI_inv
            warnings.warn('Unknown mineral type, using default gold thresholds')
        
        return F_thr, delta_thr, Moran_thr, enh_func
    
    # ================= 7. Yakymchuk Resonance Parameters =================
    
    @staticmethod
    def get_yakymchuk_params(mineral_type: str) -> Dict[str, float]:
        """
        Get Yakymchuk resonance parameters for specific mineral type.
        
        Args:
            mineral_type: Type of mineral
            
        Returns:
            Dictionary with 'a', 'b', 'c' parameters
        """
        param_table = {
            'gold':             {'a': 50, 'b': 150, 'c': 20},
            'silver':           {'a': 45, 'b': 135, 'c': 19},
            'copper':           {'a': 40, 'b': 120, 'c': 18},
            'iron':             {'a': 35, 'b': 100, 'c': 15},
            'aluminum':         {'a': 48, 'b': 140, 'c': 19},
            'coal':             {'a': 32, 'b': 80, 'c': 16},
            'lead':             {'a': 42, 'b': 125, 'c': 18},
            'zinc':             {'a': 42, 'b': 125, 'c': 18},
            'nickel':           {'a': 35, 'b': 105, 'c': 16},
            'cobalt':           {'a': 38, 'b': 115, 'c': 17},
            'molybdenum':       {'a': 48, 'b': 140, 'c': 20},
            'rare_earth':       {'a': 45, 'b': 140, 'c': 18},
            'fluorite':         {'a': 55, 'b': 170, 'c': 22},
            'tin':              {'a': 52, 'b': 155, 'c': 21},
            'tungsten':         {'a': 52, 'b': 155, 'c': 21},
            'petroleum':        {'a': 30, 'b': 70, 'c': 15},
            'gas':              {'a': 28, 'b': 75, 'c': 14},
            'coalbed_gas':      {'a': 32, 'b': 80, 'c': 16},
            'helium':           {'a': 25, 'b': 85, 'c': 14},
            'lithium':          {'a': 40, 'b': 110, 'c': 17},
            'natural_hydrogen': {'a': 30, 'b': 80, 'c': 15},
            'potassium':        {'a': 45, 'b': 135, 'c': 19},
            'uranium':          {'a': 40, 'b': 130, 'c': 19},
            'cave':             {'a': 40, 'b': 120, 'c': 18},
        }
        
        mineral_type_lower = mineral_type.lower()
        
        if mineral_type_lower in param_table:
            return param_table[mineral_type_lower]
        else:
            warnings.warn('Unknown mineral type, using default gold parameters')
            return {'a': 50, 'b': 150, 'c': 20}
