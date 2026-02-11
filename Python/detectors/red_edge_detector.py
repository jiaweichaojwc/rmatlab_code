"""
Red Edge Detector for mineral anomaly detection.
Equivalent to RedEdgeDetector.m
"""
import numpy as np
from typing import Dict, Any
from .anomaly_detector import AnomalyDetector
from ..utils.geo_utils import GeoUtils


class RedEdgeDetector(AnomalyDetector):
    """
    Red Edge Position based anomaly detector.
    
    Detects anomalies using:
    - S2REP (Sentinel-2 Red Edge Position) calculation
    - F_map (normalized anomaly intensity)
    - delta_red_edge (shift from center wavelength)
    - Moran's I local spatial autocorrelation
    - Z-score normalization and threshold-based masking
    """
    
    def calculate(self, ctx: Any) -> Dict[str, Any]:
        """
        Calculate red edge based anomaly detection.
        
        Args:
            ctx: Data context object containing:
                - s2: Sentinel-2 data cube (H x W x bands)
                - inROI: Boolean ROI mask (H x W)
                - mineral_type: Type of mineral for threshold selection
                - levashov_mode: Optional flag for Levashov threshold adjustment (default: True)
        
        Returns:
            Dictionary with:
                - mask: Binary detection mask (H x W)
                - debug: Dictionary containing intermediate results:
                    - F_map: Anomaly intensity map
                    - delta_red_edge: Red edge shift from center
                    - moran_local: Local Moran's I values
        """
        # 1. Get bands (indices are 1-based in MATLAB, converted internally)
        B4 = GeoUtils.get_band(ctx.s2, 3)  # Band 3 -> Red (B4)
        B5 = GeoUtils.get_band(ctx.s2, 7)  # Band 7 -> Red Edge 1 (B5)
        B6 = GeoUtils.get_band(ctx.s2, 8)  # Band 8 -> Red Edge 2 (B6)
        B7 = GeoUtils.get_band(ctx.s2, 9)  # Band 9 -> Red Edge 3 (B7)
        
        # 2. Calculate S2REP (Red Edge Position)
        scale_factors = [1.9997e-05, 1.9998e-05, 1.9998e-05, 1.9999e-05]
        offsets = [-0.1, -0.1, -0.1, -0.1]
        S2REP, _ = GeoUtils.calculate_s2rep_from_dn(B4, B5, B6, B7, scale_factors, offsets)
        
        # 3. Calculate F_map (anomaly intensity)
        lambda_center = 705
        delta_red_edge = S2REP - lambda_center
        F_map = np.abs(delta_red_edge) / lambda_center
        
        # 4. Moran I calculation (strictly replicating untitled.m logic)
        
        # (A) Z-score statistics
        # Strict replication of untitled.m (global statistics)
        # Uses mean/std of entire F_map to match original behavior
        F_mean = np.nanmean(F_map)
        F_std = np.nanstd(F_map)
        
        if F_std == 0:
            F_std = np.finfo(float).eps
            
        Z = (F_map - F_mean) / F_std
        
        # (B) Local Sum (using GeoUtils)
        ls = GeoUtils.calc_local_sum_with_nan(Z)
        
        # (C) Raw Moran value
        moran_raw = Z * ls
        
        # (D) Normalization (replicating untitled.m's mat2gray logic)
        # Original logic normalizes over all non-NaN values
        moran_local = np.full(moran_raw.shape, np.nan, dtype=np.float32)
        valid_mask = ~np.isnan(moran_raw)
        
        if np.any(valid_mask):
            min_v = np.min(moran_raw[valid_mask])
            max_v = np.max(moran_raw[valid_mask])
            
            if max_v - min_v < np.finfo(float).eps:
                moran_local[valid_mask] = 0
            else:
                moran_local[valid_mask] = (moran_raw[valid_mask] - min_v) / (max_v - min_v)
        
        # Clean invalid regions (outside ROI set to 0)
        moran_local[~ctx.inROI] = 0
        moran_local[np.isnan(moran_local)] = 0
        
        # 5. Get thresholds and apply Levashov correction
        F_thr, delta_thr, Moran_thr, _ = GeoUtils.get_mineral_thresholds(ctx.mineral_type)
        
        # ==========================================================
        # [Core Fix] Levashov mode threshold adjustment (matching untitled.m line 120)
        # ==========================================================
        is_levashov = True  # Default to True to match original script
        if hasattr(ctx, 'levashov_mode'):
            is_levashov = ctx.levashov_mode
        
        if is_levashov:
            # Levashov enhancement: thresholds x0.8, negative threshold x1.2
            F_thr = F_thr * 0.8
            Moran_thr = Moran_thr * 0.8
            delta_thr = delta_thr * 1.2  # Negative value relaxed
        # ==========================================================
        
        # 6. Generate mask
        # Strictly replicate filtering conditions
        mask = (
            (F_map > F_thr) &
            (delta_red_edge < delta_thr) &
            (moran_local > Moran_thr) &
            ctx.inROI &
            ~np.isnan(F_map)
        )
        
        mask = mask.astype(np.float64)
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
