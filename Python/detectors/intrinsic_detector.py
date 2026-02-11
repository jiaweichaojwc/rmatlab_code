"""
Intrinsic Absorption Detector for mineral anomaly detection.
Equivalent to IntrinsicDetector.m
"""
import numpy as np
from typing import Dict, Any
from scipy.signal import convolve2d
from scipy.ndimage import binary_opening
from .anomaly_detector import AnomalyDetector
from ..utils.geo_utils import GeoUtils


class IntrinsicDetector(AnomalyDetector):
    """
    Intrinsic absorption-based anomaly detector.
    
    Detects anomalies using:
    - Intrinsic absorption features from ASTER data
    - ROI-based normalization (critical step)
    - Moran's I local spatial autocorrelation using conv2 approach
    - NaN->0 preprocessing before convolution (not general NaN-handling)
    - Max normalization (not mat2gray)
    - Dynamic thresholds based on 95% percentile
    - Levashov mode threshold adjustment (x0.8 if enabled)
    - Morphological opening for noise reduction
    """
    
    def calculate(self, ctx: Any) -> Dict[str, Any]:
        """
        Calculate intrinsic absorption-based anomaly detection.
        
        This method strictly replicates the both.m logic (Lines 383-473):
        1. Compute raw intrinsic absorption
        2. Apply mat2gray_roi normalization FIRST (critical for proper scaling)
        3. Compute Moran I using conv2 approach (NaN->0, then convolution)
        4. Normalize by max_ls (not mat2gray)
        5. Apply Levashov mode threshold adjustment if enabled (x0.8)
        6. Use 95% percentile for dynamic thresholds
        7. Apply morphological opening with 3x3 structuring element
        
        Args:
            ctx: Data context object containing:
                - ast: ASTER data cube (H x W x 14)
                - inROI: Boolean ROI mask (H x W)
                - mineral_type: Type of mineral for threshold selection
                - levashov_mode: Optional flag for Levashov threshold adjustment (default behavior: not enabled if missing)
        
        Returns:
            Dictionary with:
                - mask: Binary detection mask (H x W)
                - debug: Dictionary containing intermediate results:
                    - F_abs: Absorption feature map (normalized)
                    - moran_local: Local Moran's I values
        """
        # 1. Compute raw intrinsic absorption intensity
        # (Calling GeoUtils formula, this step is correct)
        F_abs_raw = GeoUtils.compute_intrinsic_absorption(ctx.ast, ctx.mineral_type)
        
        # [KEY STEP 1] Strictly replicate both.m Line 383
        # Must normalize to 0-1 within ROI first, otherwise subsequent values are too small
        F_abs = GeoUtils.mat2gray_roi(F_abs_raw, ctx.inROI)
        
        # 2. Compute Moran I (must use conv2 approach, not general function!)
        # Corresponding to both.m Lines 437-450
        
        # (A) Z-Score standardization
        F_vals = F_abs[ctx.inROI]
        F_mean = np.nanmean(F_vals)
        F_std = np.nanstd(F_vals)
        
        if F_std == 0:
            F_std = np.finfo(float).eps
        
        Z = (F_abs - F_mean) / F_std
        Z[~ctx.inROI] = np.nan
        
        # (B) [KEY STEP 2] NaN handling (both.m Line 441)
        # Logic here: convert NaN to 0, so they participate in convolution sum
        # (Previous general function ignored NaN, leading to different edge results)
        Z_conv = Z.copy()
        Z_conv[np.isnan(Z_conv) | np.isinf(Z_conv)] = 0
        
        # (C) [KEY STEP 3] Use conv2 convolution sum (both.m Lines 442-443)
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0  # Center pixel excluded
        local_sum = convolve2d(Z_conv, kernel, mode='same', boundary='fill', fillvalue=0)
        local_sum[np.isnan(local_sum)] = 0
        
        # (D) [KEY STEP 4] Normalization (both.m Lines 446-448)
        # Normalization here divides by max_local_sum, not mat2gray(0-1)
        ls_roi = local_sum[ctx.inROI]
        max_ls = np.nanmax(ls_roi)
        
        if max_ls is None or max_ls == 0 or np.isnan(max_ls):
            max_ls = np.finfo(float).eps
        
        moran_local = Z * local_sum / max_ls
        
        # (E) Clean invalid values
        moran_local[~ctx.inROI] = np.nan
        moran_local[np.isnan(moran_local) | np.isinf(moran_local)] = 0
        
        # 3. Dynamic threshold calculation (95% percentile)
        # Corresponding to both.m Lines 453-463
        F_thr_base, _, Moran_thr_base, _ = GeoUtils.get_mineral_thresholds(ctx.mineral_type)
        
        # =========================================================
        # [NEW] Levashov mode judgment (replicate both.m Lines 67-72)
        # If weak signal enhancement is enabled, thresholds reduced by 20%
        # =========================================================
        if hasattr(ctx, 'levashov_mode') and ctx.levashov_mode:
            F_thr_base = F_thr_base * 0.8
            Moran_thr_base = Moran_thr_base * 0.8
        # =========================================================
        
        # Extract valid ROI values for statistics
        F_roi_vals = F_abs[ctx.inROI & ~np.isnan(F_abs)]
        m_roi_vals = moran_local[ctx.inROI & ~np.isnan(moran_local)]
        
        if len(F_roi_vals) == 0:
            F_dyn = 0
        else:
            F_dyn = np.percentile(F_roi_vals, 95)
        
        if len(m_roi_vals) == 0:
            M_dyn = 0
        else:
            M_dyn = np.percentile(m_roi_vals, 95)
        
        # Merge thresholds
        F_final = max(F_thr_base, F_dyn * 0.9)
        M_final = max(Moran_thr_base, M_dyn * 0.9)
        
        # 4. Generate mask
        cond_F = F_abs > F_final
        cond_M = moran_local > M_final
        cond_Valid = ~np.isnan(F_abs) & ~np.isinf(F_abs) & ~np.isnan(moran_local)
        
        mask = cond_F & cond_M & cond_Valid & ctx.inROI
        mask = mask.astype(np.float64)
        
        # 5. Morphological noise reduction (both.m Line 473)
        # Use 3x3 structuring element for opening operation
        se = np.ones((3, 3), dtype=bool)
        mask = binary_opening(mask.astype(bool), structure=se).astype(np.float64)
        
        # Return results
        res = {
            'mask': mask,
            'debug': {
                'F_abs': F_abs,
                'moran_local': moran_local
            }
        }
        
        return res
