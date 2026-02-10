"""
Red Edge Detector for mineral anomaly detection.
红边探测器 - 用于矿产异常探测
"""

import numpy as np
from .anomaly_detector import AnomalyDetector
from ..utils.geo_utils import GeoUtils


class RedEdgeDetector(AnomalyDetector):
    """
    Detector based on red edge position analysis.
    基于红边位置分析的探测器
    """
    
    def calculate(self, ctx) -> dict:
        """
        Calculate red edge anomalies.
        
        Parameters:
        -----------
        ctx : GeoDataContext
            Data context containing all input data
            
        Returns:
        --------
        dict
            Result dictionary with 'mask' and 'debug' keys
        """
        # 1. Get bands
        B4 = GeoUtils.get_band(ctx.s2, 3)
        B5 = GeoUtils.get_band(ctx.s2, 7)
        B6 = GeoUtils.get_band(ctx.s2, 8)
        B7 = GeoUtils.get_band(ctx.s2, 9)
        
        # 2. Calculate S2REP (Red Edge Position)
        scale_factors = [1.9997e-05, 1.9998e-05, 1.9998e-05, 1.9999e-05]
        offsets = [-0.1, -0.1, -0.1, -0.1]
        S2REP, _ = GeoUtils.calculate_s2rep_from_dn(B4, B5, B6, B7, scale_factors, offsets)
        
        # 3. Calculate anomaly intensity F_map
        lambda_center = 705
        delta_red_edge = S2REP - lambda_center
        F_map = np.abs(delta_red_edge) / lambda_center
        
        # 4. Moran I calculation
        # (A) Z-score normalization (using global statistics to match MATLAB version)
        F_mean = np.nanmean(F_map)
        F_std = np.nanstd(F_map)
        
        if F_std == 0:
            F_std = np.finfo(float).eps
        
        Z = (F_map - F_mean) / F_std
        
        # (B) Local Sum calculation
        ls = GeoUtils.calc_local_sum_with_nan(Z)
        
        # (C) Raw Moran value
        moran_raw = Z * ls
        
        # (D) Normalize to [0, 1]
        moran_local = np.full_like(moran_raw, np.nan)
        valid_mask = ~np.isnan(moran_raw)
        
        if np.any(valid_mask):
            min_v = np.min(moran_raw[valid_mask])
            max_v = np.max(moran_raw[valid_mask])
            
            if max_v - min_v < np.finfo(float).eps:
                moran_local[valid_mask] = 0
            else:
                moran_local[valid_mask] = (moran_raw[valid_mask] - min_v) / (max_v - min_v)
        
        # Clean invalid regions (outside ROI)
        moran_local[~ctx.inROI] = 0
        moran_local[np.isnan(moran_local)] = 0
        
        # 5. Get thresholds and apply Levashov correction
        F_thr, delta_thr, Moran_thr, _ = GeoUtils.get_mineral_thresholds(ctx.mineral_type)
        
        # Levashov mode: reduce thresholds for weak signal enhancement
        is_levashov = getattr(ctx, 'levashov_mode', True)
        
        if is_levashov:
            F_thr = F_thr * 0.8
            Moran_thr = Moran_thr * 0.8
            delta_thr = delta_thr * 1.2  # Relax negative threshold
        
        # 6. Generate mask
        mask = ((F_map > F_thr) & 
                (delta_red_edge < delta_thr) & 
                (moran_local > Moran_thr) & 
                ctx.inROI & 
                ~np.isnan(F_map))
        
        mask = mask.astype(float)
        mask[np.isnan(mask)] = 0
        
        # 7. Return results
        res = {
            'mask': mask,
            'debug': {
                'F_map': F_map,
                'delta_red_edge': delta_red_edge,
                'moran_local': moran_local
            }
        }
        
        return res
