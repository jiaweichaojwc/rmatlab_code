"""
GeoUtils - Geospatial utility functions for satellite data processing and mineral detection.

This module provides comprehensive tools for:
- Region configuration and ROI extraction
- Satellite data reading (Sentinel-2, Landsat-8, ASTER)
- DEM processing and terrain analysis
- Spectral indices calculation
- Mineral detection thresholds and parameters
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import generic_filter
from shapely.geometry import Polygon, Point
from typing import Tuple, List, Optional, Dict, Any, Callable
import warnings
from pathlib import Path

# Try to import tkinter, make it optional
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None
    filedialog = None
    warnings.warn("tkinter not available. GUI file selection dialogs will not work. "
                 "Install python3-tk package to enable this functionality.")


class GeoUtils:
    """Static methods for geospatial data processing and analysis."""

    # ================= 1. Region Configuration and Path Retrieval =================

    @staticmethod
    def get_region_config(region_type: str = None) -> Tuple[str, str, np.ndarray, np.ndarray]:
        """
        Interactive mode: Force manual selection of data directory and ROI file.

        Args:
            region_type: Optional region type identifier (currently unused, for future expansion)

        Returns:
            Tuple containing:
                - data_dir: Selected data directory path
                - roi_file: Selected ROI file absolute path
                - belt_lon: Longitude coordinates of bounding box (closed polygon)
                - belt_lat: Latitude coordinates of bounding box (closed polygon)

        Raises:
            ValueError: If user cancels selection
            ImportError: If tkinter is not available
        """
        if not TKINTER_AVAILABLE:
            raise ImportError("tkinter is not available. Cannot use interactive file selection. "
                            "Please install python3-tk package or provide file paths directly.")

        # Hide the main tkinter window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Step 1: Select data folder
        print('>>> [Interactive Mode] 1. Please select the data folder...')
        data_dir = filedialog.askdirectory(
            title='Please select the data folder',
            initialdir=os.getcwd()
        )
        if not data_dir:
            raise ValueError('User canceled folder selection, program terminated.')

        # Default starting path for ROI file
        default_path = os.path.dirname(data_dir)

        # Step 2: Select coordinate file
        print('>>> [Interactive Mode] 2. Please select the longitude/latitude coordinate Excel file...')
        roi_file = filedialog.askopenfilename(
            title='Please select the coordinate file',
            initialdir=default_path,
            filetypes=[
                ('Excel files', '*.xlsx *.xls'),
                ('CSV files', '*.csv'),
                ('All files', '*.*')
            ]
        )
        if not roi_file:
            raise ValueError('User canceled coordinate file selection, program terminated.')

        print(f'    ✅ Selected coordinate file: {roi_file}')

        # Get mineralization belt range
        belt_lon, belt_lat = GeoUtils.get_belt_coords(os.path.dirname(roi_file), roi_file)

        root.destroy()
        return data_dir, roi_file, belt_lon, belt_lat

    @staticmethod
    def get_belt_coords(root_dir: str, roi_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mineralization belt coordinates from ROI file.

        Args:
            root_dir: Root directory path
            roi_file: ROI file path or filename

        Returns:
            Tuple of (belt_lon, belt_lat) arrays forming a closed rectangular polygon
        """
        # Combine full path if needed
        if os.path.isfile(roi_file):
            fullpath_roi = roi_file
        else:
            fullpath_roi = os.path.join(root_dir, roi_file)

        # Use robust reading to get valid coordinates
        _, _, _, _, lon_valid, lat_valid = GeoUtils.read_roi_robust(fullpath_roi)

        # Construct closed rectangular frame
        x1 = round(np.min(lon_valid), 4)
        x2 = round(np.max(lon_valid), 4)
        y1 = round(np.min(lat_valid), 4)
        y2 = round(np.max(lat_valid), 4)

        belt_lon = np.array([x1, x2, x2, x1, x1])
        belt_lat = np.array([y1, y1, y2, y2, y1])

        print(f'    Auto-identified coordinate range: Lon[{x1:.2f} ~ {x2:.2f}], '
              f'Lat[{y1:.2f} ~ {y2:.2f}]')

        return belt_lon, belt_lat

    # ================= 2. Intelligent ROI Reading Function =================

    @staticmethod
    def read_roi_robust(fullpath_roi: str) -> Tuple[Polygon, Optional[np.ndarray],
                                                      Optional[np.ndarray], Optional[np.ndarray],
                                                      np.ndarray, np.ndarray]:
        """
        Robust reading of ROI coordinates from Excel/CSV file with intelligent column detection.

        Args:
            fullpath_roi: Full path to ROI coordinate file

        Returns:
            Tuple containing:
                - roi_poly: Shapely Polygon of ROI
                - in_roi_vec: None (placeholder for compatibility)
                - lon_grid: None (placeholder for compatibility)
                - lat_grid: None (placeholder for compatibility)
                - lon_roi: Array of longitude coordinates
                - lat_roi: Array of latitude coordinates

        Raises:
            ValueError: If coordinate columns cannot be identified
            Exception: If file reading fails
        """
        try:
            # 1. Read all raw content
            file_ext = os.path.splitext(fullpath_roi)[1].lower()
            if file_ext == '.csv':
                raw_data = pd.read_csv(fullpath_roi, header=None)
            else:  # Excel files
                raw_data = pd.read_excel(fullpath_roi, header=None, engine='openpyxl')

            rows, cols = raw_data.shape

            # 2. Intelligently identify longitude/latitude columns
            candidates_lon = []
            candidates_lat = []
            parsed_cols = {}

            for c in range(cols):
                col_data = raw_data.iloc[:, c]
                nums = np.full(rows, np.nan)
                valid_count = 0

                for r in range(rows):
                    val = col_data.iloc[r]
                    try:
                        if pd.notna(val):
                            if isinstance(val, (int, float)):
                                nums[r] = float(val)
                                valid_count += 1
                            elif isinstance(val, str):
                                num_val = float(val)
                                nums[r] = num_val
                                valid_count += 1
                    except (ValueError, TypeError):
                        nums[r] = np.nan

                if valid_count > 3:
                    valid_nums = nums[~np.isnan(nums)]
                    mean_val = np.mean(valid_nums)
                    min_val = np.min(valid_nums)
                    max_val = np.max(valid_nums)
                    parsed_cols[c] = nums

                    # Feature-based judgment
                    # Longitude: typically 60-160 for China, range < 20
                    if 60 < mean_val < 160 and (max_val - min_val) < 20:
                        candidates_lon.append(c)
                    # Latitude: typically 0-60 for China, range < 20
                    if 0 < mean_val < 60 and (max_val - min_val) < 20:
                        candidates_lat.append(c)

            # 3. Decision logic
            if len(candidates_lon) == 1 and len(candidates_lat) == 1:
                final_lon_col = candidates_lon[0]
                final_lat_col = candidates_lat[0]
            else:
                # Default: 2nd column Lon, 3rd column Lat (if intelligent detection fails)
                if cols >= 3 and 1 in parsed_cols:
                    final_lon_col = 1
                    final_lat_col = 2
                elif cols >= 2 and 0 in parsed_cols:
                    final_lon_col = 0
                    final_lat_col = 1
                else:
                    raise ValueError('Cannot identify longitude/latitude columns, please check file format.')

            lon_roi = parsed_cols[final_lon_col]
            lat_roi = parsed_cols[final_lat_col]

            # Filter valid coordinates
            valid_mask = ~np.isnan(lon_roi) & ~np.isnan(lat_roi)
            lon_roi = lon_roi[valid_mask]
            lat_roi = lat_roi[valid_mask]

            if len(lon_roi) == 0:
                raise ValueError('No valid coordinate data found')

            # Close the polygon if not already closed
            if abs(lon_roi[0] - lon_roi[-1]) > 1e-6 or abs(lat_roi[0] - lat_roi[-1]) > 1e-6:
                lon_roi = np.append(lon_roi, lon_roi[0])
                lat_roi = np.append(lat_roi, lat_roi[0])

            # Create Shapely polygon
            coords = list(zip(lon_roi, lat_roi))
            roi_poly = Polygon(coords)

            # Placeholders for compatibility with MATLAB version
            in_roi_vec = None
            lon_grid = None
            lat_grid = None

            return roi_poly, in_roi_vec, lon_grid, lat_grid, lon_roi, lat_roi

        except Exception as e:
            print(f'Error reading coordinate file: {str(e)}')
            raise

    # ================= 3. Data Reading Wrappers =================

    @staticmethod
    def read_sentinel2(data_dir: str) -> Tuple[np.ndarray, Any, str]:
        """
        Read Sentinel-2 L2A data.

        Args:
            data_dir: Data directory path

        Returns:
            Tuple containing:
                - s2: Sentinel-2 data cube (H x W x 9) with reflectance values (0-1)
                - R: Rasterio transform object
                - ref_tif_path: Reference TIFF file path

        Raises:
            FileNotFoundError: If no Sentinel-2 files found
        """
        # Search for Sentinel-2 directory
        s2_patterns = ['Sentinel*2 L2*', 'Sentinel*2*L2*', 'Sentinel-2*']
        s2_dir = None
        for pattern in s2_patterns:
            matches = list(Path(data_dir).glob(pattern))
            if matches:
                s2_dir = str(matches[0])
                break

        if s2_dir is None:
            raise FileNotFoundError('No Sentinel-2 L2A directory found')

        # Find files - prefer B08 band
        file_patterns = ['*B08*.tif', '*B08*.jp2', '*.tif*', '*.jp2']
        files = []
        for pattern in file_patterns:
            files = list(Path(s2_dir).glob(pattern))
            if files:
                break

        if not files:
            raise FileNotFoundError('No Sentinel-2 L2A files found')

        first_file = str(files[0])
        ref_tif_path = first_file

        # Read reference raster
        with rasterio.open(first_file) as src:
            R = src.transform
            ref_shape = src.shape
            ref_bounds = src.bounds
            ref_crs = src.crs

        # Band patterns for Sentinel-2
        s2_patterns = [
            ['B02'],  # Blue
            ['B03'],  # Green
            ['B04'],  # Red
            ['B08'],  # NIR
            ['B11'],  # SWIR1
            ['B12'],  # SWIR2
            ['B05'],  # Red Edge 1
            ['B06'],  # Red Edge 2
            ['B07']   # Red Edge 3
        ]

        s2_raw = GeoUtils.read_multi_bands_smart(s2_dir, s2_patterns, R, ref_shape, 9)
        s2 = s2_raw.astype(np.float32) * 0.0001  # Convert DN to reflectance

        return s2, R, ref_tif_path

    @staticmethod
    def read_landsat8(data_dir: str, R: Any, ref_shape: Tuple[int, int]) -> np.ndarray:
        """
        Read Landsat 8 data (L1 or L2).

        Args:
            data_dir: Data directory path
            R: Reference raster transform
            ref_shape: Reference raster shape (H, W)

        Returns:
            Landsat 8 data cube (H x W x 7)

        Raises:
            FileNotFoundError: If no Landsat 8 data found
        """
        # Search for Landsat 8 directories
        lan_patterns = ['*Landsat*8*L1*', '*Landsat*8*L2*', '*LC08*']
        lan_dir = None
        for pattern in lan_patterns:
            matches = list(Path(data_dir).glob(pattern))
            if matches:
                lan_dir = str(matches[0])
                break

        if lan_dir is None:
            raise FileNotFoundError('No Landsat 8 data found')

        # Band patterns for Landsat 8
        lan_patterns = [
            ['B2'],  # Blue
            ['B3'],  # Green
            ['B4'],  # Red
            ['B5'],  # NIR
            ['B6'],  # SWIR1
            ['B7'],  # SWIR2
            ['B8']   # Pan
        ]

        lan = GeoUtils.read_multi_bands_smart(lan_dir, lan_patterns, R, ref_shape, 7)
        return lan

    @staticmethod
    def read_aster(data_dir: str, R: Any, ref_shape: Tuple[int, int]) -> np.ndarray:
        """
        Read ASTER data (L2 or L1).

        Args:
            data_dir: Data directory path
            R: Reference raster transform
            ref_shape: Reference raster shape (H, W)

        Returns:
            ASTER data cube (H x W x 14) with calibrated values

        Raises:
            FileNotFoundError: If no ASTER data found
        """
        # Search for ASTER directories
        aster_patterns = ['*ASTER*L2*', '*ASTER*L1*', '*AST_*']
        aster_dir = None
        for pattern in aster_patterns:
            matches = list(Path(data_dir).glob(pattern))
            if matches:
                aster_dir = str(matches[0])
                break

        if aster_dir is None:
            raise FileNotFoundError('No ASTER data found')

        # Band patterns for ASTER (14 bands: VNIR 1-3N, SWIR 4-9, TIR 10-14)
        aster_pat = [
            ['B01', 'B1'],
            ['B02', 'B2'],
            ['B3N', 'B03N'],
            ['B04', 'B4'],
            ['B05', 'B5'],
            ['B06', 'B6'],
            ['B07', 'B7'],
            ['B08', 'B8'],
            ['B09'],
            ['B10'],
            ['B11'],
            ['B12'],
            ['B13'],
            ['B14']
        ]

        H, W = ref_shape
        ast = np.full((H, W, 14), np.nan, dtype=np.float32)

        for b in range(14):
            single_band = GeoUtils.read_any_smart(aster_dir, aster_pat[b], R, ref_shape)

            # Apply calibration
            if b <= 8:  # VNIR and SWIR bands (0-8, i.e., B1-B9)
                single_band = single_band * 0.01
                single_band[np.isinf(single_band)] = np.nan
            else:  # TIR bands (9-13, i.e., B10-B14)
                single_band = single_band * 0.1 + 300
                single_band[np.isinf(single_band)] = 300

            ast[:, :, b] = single_band

        return ast

    @staticmethod
    def read_dem_and_roi(data_dir: str, roi_file: str, R: Any,
                         ref_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray]:
        """
        Read DEM and ROI data, create coordinate grids.

        Args:
            data_dir: Data directory path
            roi_file: ROI file path (can be absolute or relative)
            R: Reference raster transform
            ref_shape: Reference raster shape (H, W)

        Returns:
            Tuple containing:
                - dem: Digital Elevation Model (H x W)
                - in_roi: Boolean mask of ROI (H x W)
                - lon_grid: Longitude grid (H x W)
                - lat_grid: Latitude grid (H x W)
                - lon_roi: ROI longitude coordinates
                - lat_roi: ROI latitude coordinates
        """
        H, W = ref_shape

        # Create coordinate grids from transform
        # Rasterio uses GDAL convention: top-left corner, y increases downward
        # R.c = x_min, R.f = y_max (top), R.a = pixel width, R.e = pixel height (negative)
        x_min = R.c
        y_max = R.f
        x_max = x_min + R.a * W
        y_min = y_max + R.e * H  # R.e is typically negative

        lon_vec = np.linspace(x_min, x_max, W)
        # Create lat_vec from top (y_max) to bottom (y_min) to match raster rows
        lat_vec = np.linspace(y_max, y_min, H)
        lon_grid, lat_grid = np.meshgrid(lon_vec, lat_vec)

        # Handle absolute or relative ROI file path
        if os.path.isfile(roi_file):
            fullpath = roi_file
        else:
            root_dir = os.path.dirname(data_dir)
            fullpath = os.path.join(root_dir, roi_file)

        # Call intelligent reading
        roi_poly, _, _, _, lon_roi, lat_roi = GeoUtils.read_roi_robust(fullpath)

        # Create ROI mask
        # Note: No flipud needed because lat_vec is already ordered top-to-bottom
        in_roi_vec = np.array([roi_poly.contains(Point(lon, lat))
                               for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())])
        in_roi = in_roi_vec.reshape(H, W)

        # Read DEM if available
        dem_patterns = ['DEM.tif', 'DEM.tiff', 'dem.tif', 'dem.tiff']
        dem_file = None
        for pattern in dem_patterns:
            dem_path = os.path.join(data_dir, pattern)
            if os.path.isfile(dem_path):
                dem_file = dem_path
                break

        if dem_file:
            with rasterio.open(dem_file) as src:
                dem_raw = src.read(1)
                dem = dem_raw.astype(np.float32)
                dem[np.isinf(dem)] = np.nan

                # Resample if needed
                if dem.shape != (H, W):
                    dem = GeoUtils._resample_to_reference(dem, src.transform, src.shape,
                                                          R, ref_shape)
        else:
            dem = np.full((H, W), np.nan, dtype=np.float32)

        return dem, in_roi, lon_grid, lat_grid, lon_roi, lat_roi

    # ================= Helper Functions =================

    @staticmethod
    def read_multi_bands_smart(dir_path: str, patterns_cell: List[List[str]],
                               R: Any, ref_shape: Tuple[int, int],
                               num_bands: int) -> np.ndarray:
        """
        Read multiple bands from a directory with smart pattern matching.

        Args:
            dir_path: Directory containing band files
            patterns_cell: List of pattern lists for each band
            R: Reference raster transform
            ref_shape: Reference raster shape (H, W)
            num_bands: Number of bands to read

        Returns:
            Data cube (H x W x num_bands)
        """
        H, W = ref_shape
        cube = np.full((H, W, num_bands), np.nan, dtype=np.float32)

        # Find all image files
        file_patterns = ['*.tif', '*.tiff', '*.jp2', '*.TIF', '*.TIFF', '*.JP2']
        files = []
        for pattern in file_patterns:
            files.extend(Path(dir_path).glob(pattern))

        for b in range(num_bands):
            patterns = patterns_cell[b]
            found = False

            for file_path in files:
                fname_upper = file_path.name.upper()
                # Check if any pattern matches
                if any(p.upper() in fname_upper for p in patterns):
                    try:
                        with rasterio.open(str(file_path)) as src:
                            A = src.read(1).astype(np.float32)

                            # Resample if dimensions don't match
                            if A.shape != (H, W):
                                A = GeoUtils._resample_to_reference(A, src.transform, src.shape,
                                                                    R, ref_shape)

                            cube[:, :, b] = A
                            found = True
                            break
                    except Exception as e:
                        warnings.warn(f'Error reading {file_path}: {str(e)}')
                        continue

            if not found:
                warnings.warn(f'Band {b+1} not found with patterns {patterns}')

        return cube

    @staticmethod
    def read_any_smart(dir_path: str, patterns_cell: List[str], R: Any,
                      ref_shape: Tuple[int, int]) -> np.ndarray:
        """
        Read a single band with smart pattern matching.

        Args:
            dir_path: Directory containing band files
            patterns_cell: List of patterns for the band
            R: Reference raster transform
            ref_shape: Reference raster shape (H, W)

        Returns:
            Single band array (H x W)
        """
        if isinstance(patterns_cell, str):
            patterns_cell = [patterns_cell]

        cube = GeoUtils.read_multi_bands_smart(dir_path, [patterns_cell], R, ref_shape, 1)
        return cube[:, :, 0]

    @staticmethod
    def _resample_to_reference(data: np.ndarray, src_transform: Any,
                               src_shape: Tuple[int, int], dst_transform: Any,
                               dst_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resample data to reference grid using bilinear interpolation.

        Args:
            data: Source data array
            src_transform: Source raster transform
            src_shape: Source shape (H, W)
            dst_transform: Destination raster transform
            dst_shape: Destination shape (H, W)

        Returns:
            Resampled data array
        """
        # Create coordinate grids
        src_h, src_w = src_shape
        dst_h, dst_w = dst_shape

        # Source coordinates
        x_src_min = src_transform.c
        y_src_max = src_transform.f
        x_src_max = x_src_min + src_transform.a * src_w
        y_src_min = y_src_max + src_transform.e * src_h

        # Destination coordinates
        x_dst_min = dst_transform.c
        y_dst_max = dst_transform.f
        x_dst_max = x_dst_min + dst_transform.a * dst_w
        y_dst_min = y_dst_max + dst_transform.e * dst_h

        # Create interpolator
        lon_src = np.linspace(x_src_min, x_src_max, src_w)
        lat_src = np.linspace(y_src_max, y_src_min, src_h)

        lon_dst = np.linspace(x_dst_min, x_dst_max, dst_w)
        lat_dst = np.linspace(y_dst_max, y_dst_min, dst_h)

        # Use scipy interpolation
        try:
            interp = RectBivariateSpline(lat_src, lon_src, data, kx=1, ky=1)
            result = interp(lat_dst, lon_dst)
            result = result.astype(np.float32)
            # Preserve NaN values in result where source had NaN
            if np.any(np.isnan(data)):
                result[np.isnan(result)] = np.nan
            return result
        except Exception:
            # Fallback to griddata
            lon_grid_src, lat_grid_src = np.meshgrid(lon_src, lat_src)
            lon_grid_dst, lat_grid_dst = np.meshgrid(lon_dst, lat_dst)

            valid_mask = ~np.isnan(data)
            if valid_mask.sum() == 0:
                return np.full(dst_shape, np.nan, dtype=np.float32)

            points = np.column_stack([lon_grid_src[valid_mask].ravel(),
                                     lat_grid_src[valid_mask].ravel()])
            values = data[valid_mask].ravel()
            result = griddata(points, values,
                            (lon_grid_dst, lat_grid_dst),
                            method='linear', fill_value=np.nan)
            return result.astype(np.float32)

    @staticmethod
    def get_band(*args) -> np.ndarray:
        """
        Get a specific band from multiple data cubes, returning the first valid one.

        Args:
            *args: Variable arguments where last argument is band index (1-based),
                   preceding arguments are data cubes to search

        Returns:
            Band array (H x W)
        """
        idx = args[-1] - 1  # Convert to 0-based index
        cubes = args[:-1]

        for cube in cubes:
            if cube.ndim == 3 and cube.shape[2] > idx:
                band = cube[:, :, idx]
                if np.count_nonzero(~np.isnan(band) & (band != 0)) > 100:
                    return band

        # Return NaN array if no valid band found
        H, W = cubes[0].shape[:2]
        return np.full((H, W), np.nan, dtype=np.float32)

    @staticmethod
    def mat2gray_roi(img: np.ndarray, in_roi: np.ndarray,
                     min_val: Optional[float] = None,
                     max_val: Optional[float] = None) -> np.ndarray:
        """
        Normalize image to [0, 1] range within ROI.

        Args:
            img: Input image
            in_roi: Boolean ROI mask
            min_val: Optional minimum value for normalization
            max_val: Optional maximum value for normalization

        Returns:
            Normalized image (values outside ROI are NaN)
        """
        img_norm = np.full(img.shape, np.nan, dtype=np.float32)
        img_roi = img[in_roi]
        img_roi = img_roi[~np.isnan(img_roi) & ~np.isinf(img_roi)]

        if len(img_roi) == 0:
            return img_norm

        if min_val is None:
            min_val = np.min(img_roi)
        if max_val is None:
            max_val = np.max(img_roi)

        if max_val - min_val < np.finfo(float).eps:
            img_norm[in_roi] = 0.5
        else:
            val = (img[in_roi] - min_val) / (max_val - min_val)
            val = np.clip(val, 0, 1)
            img_norm[in_roi] = val

        return img_norm

    @staticmethod
    def calculate_s2rep_from_dn(B4: np.ndarray, B5: np.ndarray, B6: np.ndarray,
                                B7: np.ndarray, scale_factors: List[float],
                                offsets: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Sentinel-2 Red Edge Position (S2REP) from DN values.

        Args:
            B4: Band 4 (Red) reflectance values (0-1 range, typically from DN*0.0001)
            B5: Band 5 (Red Edge 1) reflectance values
            B6: Band 6 (Red Edge 2) reflectance values
            B7: Band 7 (Red Edge 3) reflectance values
            scale_factors: Scale factors for each band [B4, B5, B6, B7]
            offsets: Offset values for each band [B4, B5, B6, B7]

        Returns:
            Tuple containing:
                - S2REP: Red Edge Position (680-760 nm)
                - REP_QA: Quality assessment flags (0=invalid, 1=valid, 2=zero_denom, 3=invalid_reflect, 4=out_of_range)

        Note:
            The input bands should be reflectance values (0-1 range).
            The formula multiplies by 10000 to convert to DN before applying scale factors,
            matching the original MATLAB implementation for compatibility.
        """
        # Convert to DN-like values and apply calibration (matches MATLAB implementation)
        B4_val = (B4 * 10000) * scale_factors[0] + offsets[0]
        B5_val = (B5 * 10000) * scale_factors[1] + offsets[1]
        B6_val = (B6 * 10000) * scale_factors[2] + offsets[2]
        B7_val = (B7 * 10000) * scale_factors[3] + offsets[3]

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
        REP_QA[invalid_reflect] = 3

        valid_pixel = ~invalid_reflect

        # Calculate S2REP
        numerator = ((B4_val + B7_val) / 2) - B5_val
        denominator = (B6_val - B5_val) + 1e-8

        # Check for zero denominator
        zero_denominator = valid_pixel & (np.abs(denominator) < 1e-6)
        REP_QA[zero_denominator] = 2
        valid_pixel = valid_pixel & ~zero_denominator

        # Calculate REP
        S2REP[valid_pixel] = 705 + 35 * (numerator[valid_pixel] / denominator[valid_pixel])

        # Check for out-of-range values
        rep_out_range = valid_pixel & ((S2REP < 680) | (S2REP > 760))
        REP_QA[rep_out_range] = 4
        S2REP[rep_out_range] = np.nan
        REP_QA[valid_pixel & ~rep_out_range] = 1

        return S2REP, REP_QA

    @staticmethod
    def compute_intrinsic_absorption(ast: np.ndarray, mineral_type: str) -> np.ndarray:
        """
        Compute intrinsic absorption features for mineral detection.

        Args:
            ast: ASTER data cube (H x W x 14)
            mineral_type: Type of mineral ('gold' or other)

        Returns:
            Absorption feature map (H x W)
        """
        eps_val = 1e-6
        H, W, _ = ast.shape
        F_abs = np.full((H, W), np.nan, dtype=np.float32)

        if mineral_type.lower() == 'gold':
            # Gold-specific absorption calculation
            cont = (ast[:, :, 2] + ast[:, :, 4]) / 2  # B3N and B5
            target = ast[:, :, 2]  # B3N
            F_abs = (cont - target) / (cont + eps_val)
            F_abs = F_abs + 0.5 * (ast[:, :, 5] / (ast[:, :, 4] + eps_val))  # B6 / B5
        else:
            # General mineral absorption
            cont = (ast[:, :, 4] + ast[:, :, 6]) / 2  # B5 and B7
            target = ast[:, :, 5]  # B6
            F_abs = (cont - target) / (cont + eps_val)

        F_abs[np.isinf(F_abs)] = np.nan
        return F_abs

    @staticmethod
    def compute_dem_indices(dem: np.ndarray, mineral_type: str, H: int, W: int,
                           in_roi: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute DEM-derived indices (slope, curvature) for mineral detection.

        Args:
            dem: Digital Elevation Model
            mineral_type: Type of mineral
            H: Height of output
            W: Width of output
            in_roi: ROI mask

        Returns:
            Dictionary with 'slope' and 'neg_curvature' arrays
        """
        indices = {
            'slope': np.full((H, W), np.nan, dtype=np.float32),
            'neg_curvature': np.full((H, W), np.nan, dtype=np.float32)
        }

        if mineral_type.lower() != 'cave':
            return indices

        # Calculate gradients
        dy, dx = np.gradient(dem)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
        indices['slope'] = GeoUtils.mat2gray_roi(slope, in_roi)

        # Calculate curvature
        dxx = np.gradient(dx, axis=1)
        dyy = np.gradient(dy, axis=0)
        curvature = -(dxx + dyy)
        neg_curv = np.maximum(-curvature, 0)
        indices['neg_curvature'] = GeoUtils.mat2gray_roi(neg_curv, in_roi)

        return indices

    @staticmethod
    def calc_local_sum_with_nan(Z: np.ndarray) -> np.ndarray:
        """
        Calculate local sum with NaN handling using 3x3 neighborhood.

        Args:
            Z: Input array

        Returns:
            Local sum array with NaN-aware computation
        """
        rows, cols = Z.shape
        local_sum = np.full((rows, cols), np.nan, dtype=np.float32)

        # Pad array
        Z_padded = np.pad(Z, pad_width=1, mode='constant', constant_values=np.nan)

        # Weight matrix (exclude center)
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

    @staticmethod
    def get_mineral_thresholds(mineral_type: str) -> Tuple[float, float, float, Callable]:
        """
        Get mineral detection thresholds and enhancement function.

        Args:
            mineral_type: Type of mineral

        Returns:
            Tuple containing:
                - F_thr: Ferric threshold
                - delta_thr: Delta threshold
                - Moran_thr: Moran's I threshold
                - enh_func: Enhancement function
        """
        F_thr = 0.018
        delta_thr = -2
        Moran_thr = 0.20

        def enh_func(Ferric: np.ndarray, Fe_anom: np.ndarray, OH_anom: np.ndarray,
                    Clay: np.ndarray, NDVI: np.ndarray, *args) -> np.ndarray:
            """
            Enhancement function for mineral detection.
            
            For standard minerals: Uses 5 parameters (Ferric, Fe_anom, OH_anom, Clay, NDVI)
            For cave type: Accepts additional DEM parameters (slope, neg_curvature) via *args
            """
            if len(args) >= 2:
                # Cave type with DEM indices
                slope = args[0]
                neg_curvature = args[1]
                return (0.45 * Ferric + 0.25 * Fe_anom + 0.15 * OH_anom +
                       0.10 * Clay + 0.05 * NDVI)
            else:
                # Standard type
                return (0.45 * Ferric + 0.25 * Fe_anom + 0.15 * OH_anom +
                       0.10 * Clay + 0.05 * NDVI)

        return F_thr, delta_thr, Moran_thr, enh_func

    @staticmethod
    def get_yakymchuk_params(mineral_type: str) -> Dict[str, float]:
        """
        Get Yakymchuk model parameters for mineral detection.

        Args:
            mineral_type: Type of mineral

        Returns:
            Dictionary with parameters 'a', 'b', 'c'
        """
        params = {
            'a': 50,
            'b': 150,
            'c': 20
        }
        return params
