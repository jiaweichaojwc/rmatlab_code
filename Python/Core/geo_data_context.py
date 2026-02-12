"""
GeoDataContext - Centralized geospatial data context for mineral detection.

This module provides a class that loads and manages all geospatial data including
Sentinel-2, Landsat-8, ASTER, DEM, and ROI data for mineral detection workflows.
"""

import sys
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Utils.geo_utils import GeoUtils


class GeoDataContext:
    """
    Central context for managing geospatial data for mineral detection.
    
    This class handles loading and preprocessing of multi-source satellite imagery
    (Sentinel-2, Landsat-8, ASTER) along with DEM and ROI data. It provides a
    unified interface for accessing prepared data for various detectors.
    
    Attributes:
        mineral_type (str): Type of mineral being detected
        region_type (str): Type of region being analyzed
        levashov_mode (bool): Whether to use Levashov detection mode
        kmz_path (Optional[str]): Path to KMZ file for additional context
        kmz_keywords (Optional[list]): Keywords for KMZ filtering
        data_dir (str): Root directory containing all data files
        ref_tif_path (str): Reference TIFF path for KML georeferencing
        s2 (np.ndarray): Sentinel-2 data cube
        lan (np.ndarray): Landsat-8 data cube
        ast (np.ndarray): ASTER data cube
        dem (np.ndarray): Digital Elevation Model data
        in_roi (np.ndarray): Boolean mask indicating pixels within ROI
        R (Dict): Spatial reference information
        lon_grid (np.ndarray): Longitude grid
        lat_grid (np.ndarray): Latitude grid
        lon_roi (np.ndarray): Longitude coordinates of ROI boundary
        lat_roi (np.ndarray): Latitude coordinates of ROI boundary
        belt_lon (np.ndarray): Longitude coordinates of belt region
        belt_lat (np.ndarray): Latitude coordinates of belt region
        nir (np.ndarray): Near-infrared band (combined from S2/Landsat)
        red (np.ndarray): Red band (combined from S2/Landsat)
        green (np.ndarray): Green band (combined from S2/Landsat)
        blue (np.ndarray): Blue band (combined from S2/Landsat)
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize GeoDataContext with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - mineral_type (str): Type of mineral to detect
                - region_type (str): Type of region
                - levashov_mode (bool, optional): Use Levashov mode, default True
                - kmz_path (str, optional): Path to KMZ file
                - kmz_keywords (list, optional): KMZ filtering keywords
                - data_dir (str, optional): Data directory path
                - roi_file (str, optional): ROI file path
        """
        # Basic configuration
        self.mineral_type: str = config['mineral_type']
        self.region_type: str = config['region_type']
        self.levashov_mode: bool = config.get('levashov_mode', True)
        
        # KML configuration
        self.kmz_path: Optional[str] = config.get('kmz_path')
        self.kmz_keywords: Optional[list] = config.get('kmz_keywords')
        
        # Path information
        self.data_dir: str = ""
        self.ref_tif_path: str = ""
        
        # Data arrays - initialize as None
        self.s2: Optional[np.ndarray] = None
        self.lan: Optional[np.ndarray] = None
        self.ast: Optional[np.ndarray] = None
        self.dem: Optional[np.ndarray] = None
        
        # Spatial reference and ROI
        self.in_roi: Optional[np.ndarray] = None
        self.R: Optional[Dict[str, Any]] = None
        self.lon_grid: Optional[np.ndarray] = None
        self.lat_grid: Optional[np.ndarray] = None
        self.lon_roi: Optional[np.ndarray] = None
        self.lat_roi: Optional[np.ndarray] = None
        
        # Belt coordinates
        self.belt_lon: Optional[np.ndarray] = None
        self.belt_lat: Optional[np.ndarray] = None
        
        # Band shortcuts
        self.nir: Optional[np.ndarray] = None
        self.red: Optional[np.ndarray] = None
        self.green: Optional[np.ndarray] = None
        self.blue: Optional[np.ndarray] = None
        
        # Load data
        self._load_data(config)
    
    def _load_data(self, config: Dict[str, Any]) -> None:
        """
        Load all geospatial data based on configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        # 1. Get paths (use config paths if available, otherwise interactive)
        if (config.get('data_dir') and config.get('roi_file')):
            # GUI mode: use provided paths
            self.data_dir = config['data_dir']
            roi_file = config['roi_file']
            
            # Get belt coordinates using GeoUtils
            roi_dir = os.path.dirname(roi_file)
            self.belt_lon, self.belt_lat = GeoUtils.get_belt_coords(roi_dir, roi_file)
        else:
            # Script mode: interactive selection
            data_dir_local, roi_file, self.belt_lon, self.belt_lat = \
                GeoUtils.get_region_config(self.region_type)
            self.data_dir = data_dir_local
        
        # 2. Load data
        print(f'  [Context] Loading data from: {self.data_dir}')
        
        # Read Sentinel-2
        self.s2, self.R, self.ref_tif_path = GeoUtils.read_sentinel2(self.data_dir)
        
        # Read Landsat-8
        self.lan = GeoUtils.read_landsat8(self.data_dir, self.R)
        
        # Read ASTER
        self.ast = GeoUtils.read_aster(self.data_dir, self.R)
        
        # Read DEM and ROI
        self.dem, self.in_roi, self.lon_grid, self.lat_grid, self.lon_roi, self.lat_roi = \
            GeoUtils.read_dem_and_roi(self.data_dir, roi_file, self.R)
        
        # Extract common bands
        self.nir = GeoUtils.get_band(self.s2, self.lan, idx=4)
        self.red = GeoUtils.get_band(self.s2, self.lan, idx=3)
        self.green = GeoUtils.get_band(self.s2, self.lan, idx=2)
        self.blue = GeoUtils.get_band(self.s2, self.lan, idx=1)
        
        # Fill ASTER NaN values
        self.fill_aster_nan()
    
    def fill_aster_nan(self) -> None:
        """
        Fill NaN values in ASTER data with mean values from ROI.
        
        For each band in the ASTER cube, calculates the mean value of valid
        pixels within the ROI and fills any NaN pixels within the ROI with
        this mean value. This ensures complete data coverage within the ROI.
        """
        if self.ast is None or self.in_roi is None:
            return
        
        # Iterate through each band
        for b in range(self.ast.shape[2]):
            band_data = self.ast[:, :, b].copy()
            
            # Get ROI values
            roi_vals = band_data[self.in_roi]
            
            # Calculate mean, ignoring NaN values
            mean_val = np.nanmean(roi_vals)
            
            # Fill NaN values within ROI with mean
            if not np.isnan(mean_val):
                mask = self.in_roi & np.isnan(band_data)
                band_data[mask] = mean_val
                self.ast[:, :, b] = band_data
