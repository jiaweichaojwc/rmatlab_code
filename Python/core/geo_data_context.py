"""
GeoDataContext - Data container for geospatial and satellite data processing.

This class acts as a centralized data container that holds:
- Configuration (mineral type, region type, processing modes)
- File paths and KMZ configuration
- Satellite data (Sentinel-2, Landsat-8, ASTER, DEM)
- Geographic grids and ROI information
- Spectral band data (NIR, Red, Green, Blue)
"""

from typing import Any, Optional, List
import numpy as np
from ..utils.geo_utils import GeoUtils


class GeoDataContext:
    """
    Data container class for geospatial satellite data and mineral detection processing.
    
    This class loads and manages all satellite data, DEM, ROI information, and derived
    spectral bands for mineral detection workflows. It mirrors the MATLAB GeoDataContext class.
    
    Attributes:
        mineral_type (str): Type of mineral to detect (e.g., 'gold', 'copper')
        region_type (str): Geographic region identifier
        levashov_mode (bool): Whether to use Levashov processing mode (default: True)
        kmz_path (Optional[str]): Path to KMZ file for geographic reference
        kmz_keywords (Optional[List[str]]): Keywords for KMZ processing
        data_dir (str): Root directory containing satellite data
        ref_tif_path (str): Path to reference TIFF file for KMZ georeferencing
        s2 (np.ndarray): Sentinel-2 satellite data
        lan (np.ndarray): Landsat-8 satellite data
        ast (np.ndarray): ASTER satellite data
        dem (np.ndarray): Digital Elevation Model data
        inROI (np.ndarray): Boolean mask indicating pixels within ROI
        R (Any): Spatial reference object for georeferencing
        lonGrid (np.ndarray): Longitude grid for the study area
        latGrid (np.ndarray): Latitude grid for the study area
        lonROI (np.ndarray): Longitude coordinates of ROI boundary
        latROI (np.ndarray): Latitude coordinates of ROI boundary
        belt_lon (np.ndarray): Longitude coordinates of mineralization belt
        belt_lat (np.ndarray): Latitude coordinates of mineralization belt
        NIR (np.ndarray): Near-infrared band data
        Red (np.ndarray): Red band data
        Green (np.ndarray): Green band data
        Blue (np.ndarray): Blue band data
    """
    
    def __init__(self, config: dict):
        """
        Initialize GeoDataContext with configuration and load all satellite data.
        
        This constructor:
        1. Extracts configuration parameters
        2. Retrieves region-specific paths and belt coordinates
        3. Loads all satellite data (Sentinel-2, Landsat-8, ASTER, DEM)
        4. Extracts spectral bands (NIR, Red, Green, Blue)
        5. Fills NaN values in ASTER data
        
        Args:
            config (dict): Configuration dictionary containing:
                - mineral_type (str): Required. Type of mineral to detect
                - region_type (str): Required. Geographic region identifier
                - levashov_mode (bool): Optional. Use Levashov mode (default: True)
                - kmz_path (str): Optional. Path to KMZ file
                - kmz_keywords (List[str]): Optional. KMZ processing keywords
        
        Raises:
            KeyError: If required config keys (mineral_type, region_type) are missing
            FileNotFoundError: If data files cannot be found
        """
        # Configuration
        self.mineral_type: str = config['mineral_type']
        self.region_type: str = config['region_type']
        self.levashov_mode: bool = config.get('levashov_mode', True)
        
        # KMZ configuration
        self.kmz_path: Optional[str] = config.get('kmz_path', None)
        self.kmz_keywords: Optional[List[str]] = config.get('kmz_keywords', None)
        
        # Initialize data attributes
        self.data_dir: str = None
        self.ref_tif_path: str = None
        self.s2: np.ndarray = None
        self.lan: np.ndarray = None
        self.ast: np.ndarray = None
        self.dem: np.ndarray = None
        self.inROI: np.ndarray = None
        self.R: Any = None
        self.lonGrid: np.ndarray = None
        self.latGrid: np.ndarray = None
        self.lonROI: np.ndarray = None
        self.latROI: np.ndarray = None
        self.belt_lon: np.ndarray = None
        self.belt_lat: np.ndarray = None
        self.NIR: np.ndarray = None
        self.Red: np.ndarray = None
        self.Green: np.ndarray = None
        self.Blue: np.ndarray = None
        
        # 1. Get region configuration and paths (interactive/automatic)
        data_dir_local, roi_file, self.belt_lon, self.belt_lat = \
            GeoUtils.get_region_config(self.region_type)
        self.data_dir = data_dir_local
        
        # 2. Load data
        print(f'  [Context] Loading data from: {self.data_dir}')
        
        # Load Sentinel-2 data and get reference TIFF path
        self.s2, self.R, self.ref_tif_path = GeoUtils.read_sentinel2(self.data_dir)
        
        # Load Landsat-8 data
        self.lan = GeoUtils.read_landsat8(self.data_dir, self.R, self.s2.shape[:2])
        
        # Load ASTER data
        self.ast = GeoUtils.read_aster(self.data_dir, self.R, self.s2.shape[:2])
        
        # Load DEM and ROI information
        self.dem, self.inROI, self.lonGrid, self.latGrid, self.lonROI, self.latROI = \
            GeoUtils.read_dem_and_roi(self.data_dir, roi_file, self.R, self.s2.shape[:2])
        
        # Extract spectral bands
        self.NIR = GeoUtils.get_band(self.s2, self.lan, 4)
        self.Red = GeoUtils.get_band(self.s2, self.lan, 3)
        self.Green = GeoUtils.get_band(self.s2, self.lan, 2)
        self.Blue = GeoUtils.get_band(self.s2, self.lan, 1)
        
        # Fill NaN values in ASTER data
        self.fill_aster_nan()
    
    def fill_aster_nan(self) -> None:
        """
        Fill NaN values in ASTER data with mean values from ROI.
        
        For each band in the ASTER data:
        1. Extract valid values within the ROI
        2. Calculate mean value (ignoring NaN)
        3. Fill NaN values within ROI with the mean
        
        This method modifies self.ast in place.
        """
        if self.ast is None or self.inROI is None:
            return
        
        # Iterate through each band
        for b in range(self.ast.shape[2]):
            band_data = self.ast[:, :, b].copy()
            
            # Get values within ROI
            roi_vals = band_data[self.inROI]
            
            # Calculate mean, ignoring NaN
            mean_val = np.nanmean(roi_vals)
            
            # If mean is valid, fill NaN values within ROI
            if not np.isnan(mean_val):
                mask = self.inROI & np.isnan(band_data)
                band_data[mask] = mean_val
                self.ast[:, :, b] = band_data
