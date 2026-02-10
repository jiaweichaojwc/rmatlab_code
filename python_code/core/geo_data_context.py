"""
Geographic Data Context for managing remote sensing data.
地理数据上下文 - 用于管理遥感数据
"""

import numpy as np
from typing import Optional
from ..utils.geo_utils import GeoUtils


class GeoDataContext:
    """
    Context class for managing all geographic and remote sensing data.
    用于管理所有地理和遥感数据的上下文类
    """
    
    def __init__(self, config: dict):
        """
        Initialize data context with configuration.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary containing:
            - mineral_type: Type of mineral to detect
            - region_type: Region identifier
            - levashov_mode: Enable Levashov enhancement
            - kmz_path: Path to KML/KMZ file (optional)
            - kmz_keywords: Keywords for KML filtering (optional)
            - data_dir: Data directory path
            - roi_file: ROI coordinate file path
        """
        # Configuration
        self.mineral_type = config.get('mineral_type', 'gold')
        self.region_type = config.get('region_type', '')
        self.levashov_mode = config.get('levashov_mode', True)
        
        # KML configuration
        self.kmz_path = config.get('kmz_path', '')
        self.kmz_keywords = config.get('kmz_keywords', [])
        
        # Paths
        self.data_dir = config.get('data_dir', '')
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
        
        # Load data
        self._load_data(config)
    
    def _load_data(self, config: dict):
        """
        Load all required data.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        print(f"  [Context] Loading data from: {self.data_dir}")
        
        # Note: This is a simplified version
        # Full implementation requires proper file paths
        
        try:
            # 1. Read Sentinel-2 data
            if self.data_dir:
                try:
                    self.s2, self.R, self.ref_tif_path = GeoUtils.read_sentinel2(self.data_dir)
                except Exception as e:
                    print(f"    警告: 无法读取 Sentinel-2 数据: {e}")
                    self.s2 = np.zeros((100, 100, 9), dtype=np.float32)
                    self.R = None
                    self.ref_tif_path = None
            
            # 2. Read Landsat 8 (optional)
            try:
                self.lan = GeoUtils.read_landsat8(self.data_dir, self.R)
            except:
                print("    警告: Landsat 8 数据不可用，使用默认值")
                if self.s2 is not None:
                    self.lan = np.zeros((*self.s2.shape[:2], 7), dtype=np.float32)
                else:
                    self.lan = np.zeros((100, 100, 7), dtype=np.float32)
            
            # 3. Read ASTER (optional)
            try:
                self.ast = GeoUtils.read_aster(self.data_dir, self.R)
            except:
                print("    警告: ASTER 数据不可用，使用默认值")
                if self.s2 is not None:
                    self.ast = np.zeros((*self.s2.shape[:2], 14), dtype=np.float32)
                else:
                    self.ast = np.zeros((100, 100, 14), dtype=np.float32)
            
            # 4. Read DEM and ROI
            roi_file = config.get('roi_file', '')
            if roi_file:
                try:
                    (self.dem, self.inROI, self.lonGrid, self.latGrid, 
                     self.lonROI, self.latROI) = GeoUtils.read_dem_and_roi(
                        self.data_dir, roi_file, self.R
                    )
                except Exception as e:
                    print(f"    警告: 无法读取 DEM/ROI: {e}")
                    shape = self.s2.shape[:2] if self.s2 is not None else (100, 100)
                    self.dem = np.zeros(shape, dtype=np.float32)
                    self.inROI = np.ones(shape, dtype=bool)
                    self.lonGrid = np.zeros(shape)
                    self.latGrid = np.zeros(shape)
                    self.lonROI = np.array([])
                    self.latROI = np.array([])
            else:
                shape = self.s2.shape[:2] if self.s2 is not None else (100, 100)
                self.dem = np.zeros(shape, dtype=np.float32)
                self.inROI = np.ones(shape, dtype=bool)
                self.lonGrid = np.zeros(shape)
                self.latGrid = np.zeros(shape)
                self.lonROI = np.array([])
                self.latROI = np.array([])
            
            # Extract color bands
            self.NIR = GeoUtils.get_band(self.s2, self.lan, 4)
            self.Red = GeoUtils.get_band(self.s2, self.lan, 3)
            self.Green = GeoUtils.get_band(self.s2, self.lan, 2)
            self.Blue = GeoUtils.get_band(self.s2, self.lan, 1)
            
            # Fill ASTER NaN values
            self._fill_aster_nan()
            
        except Exception as e:
            print(f"  错误: 数据加载失败: {e}")
            raise
    
    def _fill_aster_nan(self):
        """Fill NaN values in ASTER data with ROI mean."""
        if self.ast is None:
            return
        
        for b in range(self.ast.shape[2]):
            band_data = self.ast[:, :, b]
            roi_vals = band_data[self.inROI]
            mean_val = np.nanmean(roi_vals)
            
            if not np.isnan(mean_val):
                mask = self.inROI & np.isnan(band_data)
                self.ast[:, :, b][mask] = mean_val
