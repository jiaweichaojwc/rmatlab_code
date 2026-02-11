"""
GeoDataContext - Geographic data context manager
Converted from MATLAB GeoDataContext.m
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.geo_utils import GeoUtils


class GeoDataContext:
    """
    Data context for geographic information and remote sensing data
    """
    
    def __init__(self, config):
        # Configuration
        self.mineral_type = config['mineral_type']
        self.region_type = config['region_type']
        self.levashov_mode = config.get('levashov_mode', True)
        
        # KML configuration
        self.kmz_path = config.get('kmz_path', '')
        self.kmz_keywords = config.get('kmz_keywords', [])
        
        # Path information
        self.data_dir = None
        self.ref_tif_path = None
        
        # Data arrays
        self.s2 = None
        self.lan = None
        self.ast = None
        self.dem = None
        self.inROI = None
        self.R = None
        self.lonGrid = None
        self.latGrid = None
        self.lonROI = None
        self.latROI = None
        self.belt_lon = None
        self.belt_lat = None
        self.NIR = None
        self.Red = None
        self.Green = None
        self.Blue = None
        
        # Initialize data
        self._loadData(config)
    
    def _loadData(self, config):
        """Load all geographic data"""
        # 1. Get paths (interactive/automatic)
        data_dir_local, roi_file, self.belt_lon, self.belt_lat = \
            GeoUtils.getRegionConfig(self.region_type)
        self.data_dir = data_dir_local
        
        # 2. Read data
        print(f'  [Context] Loading data from: {self.data_dir}')
        
        # Read Sentinel-2 (returns ref_tif_path for KML alignment)
        self.s2, self.R, self.ref_tif_path = GeoUtils.readSentinel2(self.data_dir)
        
        # Read Landsat 8
        self.lan = GeoUtils.readLandsat8(self.data_dir, self.R)
        
        # Read ASTER
        self.ast = GeoUtils.readASTER(self.data_dir, self.R)
        
        # Read DEM and ROI
        self.dem, self.inROI, self.lonGrid, self.latGrid, self.lonROI, self.latROI = \
            GeoUtils.readDEMandROI(self.data_dir, roi_file, self.R)
        
        # Get standard bands
        self.NIR = GeoUtils.getBand(self.s2, self.lan, 4)
        self.Red = GeoUtils.getBand(self.s2, self.lan, 3)
        self.Green = GeoUtils.getBand(self.s2, self.lan, 2)
        self.Blue = GeoUtils.getBand(self.s2, self.lan, 1)
        
        # Fill ASTER NaN values
        self.fillAsterNaN()
    
    def fillAsterNaN(self):
        """Fill NaN values in ASTER data with ROI mean"""
        for b in range(self.ast.shape[2]):
            bandData = self.ast[:, :, b].copy()
            roi_vals = bandData[self.inROI]
            mean_val = np.nanmean(roi_vals)
            
            if not np.isnan(mean_val):
                mask = self.inROI & np.isnan(bandData)
                bandData[mask] = mean_val
                self.ast[:, :, b] = bandData
