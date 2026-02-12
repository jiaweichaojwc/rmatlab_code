"""
Red Edge Detector for anomaly detection based on Sentinel-2 red-edge position.
"""

import numpy as np
from typing import Dict, Any
from .anomaly_detector import AnomalyDetector
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.geo_utils import GeoUtils


class RedEdgeDetector(AnomalyDetector):
    """
    Detects anomalies using red-edge position (S2REP) from Sentinel-2 data.
    
    The red-edge is the sharp increase in reflectance between red and NIR
    wavelengths, which shifts position in the presence of certain minerals.
    """
    
    def calculate(self, ctx) -> Dict[str, Any]:
        """
        Calculate red-edge anomaly mask and intermediate results.
        
        Args:
            ctx: GeoDataContext object with loaded satellite data
            
        Returns:
            Dictionary with 'mask' (binary array) and 'debug' (intermediate results)
        """
        # 1. Get spectral bands
        B4 = GeoUtils.get_band(ctx.s2, band_index=3)
        B5 = GeoUtils.get_band(ctx.s2, band_index=7)
        B6 = GeoUtils.get_band(ctx.s2, band_index=8)
        B7 = GeoUtils.get_band(ctx.s2, band_index=9)
        
        # 2. Calculate S2REP (red edge position)
        scale_factors = [1.9997e-05, 1.9998e-05, 1.9998e-05, 1.9999e-05]
        offsets = [-0.1, -0.1, -0.1, -0.1]
        S2REP, _ = GeoUtils.calculate_s2rep_from_dn(B4, B5, B6, B7, scale_factors, offsets)
        
        # 3. Calculate anomaly strength F_map
        lambda_center = 705  # Reference wavelength (nm)
        delta_red_edge = S2REP - lambda_center
        F_map = np.abs(delta_red_edge) / lambda_center
        
        # 4. Calculate Moran I spatial autocorrelation
        
        # (A) Z-score normalization (using ROI statistics)
        F_vals = F_map[ctx.in_roi]
        F_mean = np.nanmean(F_vals)
        F_std = np.nanstd(F_vals)
        
        if F_std == 0:
            F_std = np.finfo(float).eps
        
        Z = (F_map - F_mean) / F_std
        
        # (B) Local sum calculation
        ls = GeoUtils.calc_local_sum_with_nan(Z)
        
        # (C) Raw Moran value
        moran_raw = Z * ls
        
        # (D) Normalize (replicate MATLAB's mat2gray logic)
        moran_local = np.full_like(moran_raw, np.nan)
        valid_mask = ~np.isnan(moran_raw)
        
        if np.any(valid_mask):
            min_v = np.min(moran_raw[valid_mask])
            max_v = np.max(moran_raw[valid_mask])
            
            if max_v - min_v < np.finfo(float).eps:
                moran_local[valid_mask] = 0
            else:
                moran_local[valid_mask] = (moran_raw[valid_mask] - min_v) / (max_v - min_v)
        
        # Clean up invalid regions
        moran_local[~ctx.in_roi] = 0
        moran_local[np.isnan(moran_local)] = 0
        
        # 5. Get thresholds with Levashov correction
        F_thr, delta_thr, Moran_thr, _ = GeoUtils.get_mineral_thresholds(ctx.mineral_type)
        
        # ===== Levashov mode threshold adjustment =====
        is_levashov = getattr(ctx, 'levashov_mode', True)
        
        if is_levashov:
            F_thr = F_thr * 0.8
            Moran_thr = Moran_thr * 0.8
            delta_thr = delta_thr * 1.2  # Relax negative threshold
        # =============================================
        
        # 6. Generate mask with strict filtering
        mask = ((F_map > F_thr) & 
                (delta_red_edge < delta_thr) & 
                (moran_local > Moran_thr) & 
                ctx.in_roi & 
                ~np.isnan(F_map))
        
        mask = mask.astype(float)
        mask[np.isnan(mask)] = 0
        
        # 7. Return results
        result = {
            'mask': mask,
            'debug': {
                'F_map': F_map,
                'delta_red_edge': delta_red_edge,
                'moran_local': moran_local
            }
        }
        
        return result
