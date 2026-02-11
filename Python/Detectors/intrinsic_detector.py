"""
IntrinsicDetector - Intrinsic absorption anomaly detector
Converted from MATLAB IntrinsicDetector.m
"""

import numpy as np
from scipy.ndimage import grey_opening, generate_binary_structure
from scipy.signal import convolve2d
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Detectors.anomaly_detector import AnomalyDetector
from Utils.geo_utils import GeoUtils


class IntrinsicDetector(AnomalyDetector):
    """
    Intrinsic absorption anomaly detector
    Detects anomalies based on spectral absorption features
    """
    
    def calculate(self, ctx):
        """Calculate intrinsic absorption anomalies"""
        # 1. Calculate raw intrinsic absorption intensity
        F_abs_raw = GeoUtils.computeIntrinsicAbsorption(ctx.ast, ctx.mineral_type)
        
        # Key Step 1: Normalize to 0-1 within ROI first
        # This matches both.m Line 383
        F_abs = GeoUtils.mat2gray_roi(F_abs_raw, ctx.inROI)
        
        # 2. Calculate Moran I (must use conv2, not generic function!)
        # Corresponds to both.m Line 437-450
        
        # (A) Z-Score standardization
        F_vals = F_abs[ctx.inROI]
        F_mean = np.nanmean(F_vals)
        F_std = np.nanstd(F_vals)
        if F_std == 0:
            F_std = np.finfo(float).eps
        
        Z = (F_abs - F_mean) / F_std
        Z[~ctx.inROI] = np.nan
        
        # (B) Key Step 2: NaN handling (both.m Line 441)
        # Convert NaN to 0 to participate in convolution
        Z[np.isnan(Z) | np.isinf(Z)] = 0
        
        # (C) Key Step 3: Use conv2 convolution (both.m Line 442-443)
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        local_sum = convolve2d(Z, kernel, mode='same')
        local_sum[np.isnan(local_sum)] = 0
        
        # (D) Key Step 4: Normalization (both.m Line 446-448)
        # Normalize by dividing by max_local_sum, not mat2gray(0-1)
        ls_roi = local_sum[ctx.inROI]
        max_ls = np.nanmax(ls_roi)
        if max_ls == 0 or np.isnan(max_ls):
            max_ls = np.finfo(float).eps
        
        moran_local = Z * local_sum / max_ls
        
        # (E) Clean invalid values
        moran_local[~ctx.inROI] = np.nan
        moran_local[np.isnan(moran_local) | np.isinf(moran_local)] = 0
        
        # 3. Dynamic threshold calculation (95% percentile)
        # Corresponds to both.m Line 453-463
        F_thr_base, _, Moran_thr_base, _ = GeoUtils.getMineralThresholds(ctx.mineral_type)
        
        # Levashov mode adjustment
        if hasattr(ctx, 'levashov_mode') and ctx.levashov_mode:
            F_thr_base = F_thr_base * 0.8
            Moran_thr_base = Moran_thr_base * 0.8
        
        # Extract valid values in ROI for statistics
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
        
        # Fuse thresholds
        F_final = max(F_thr_base, F_dyn * 0.9)
        M_final = max(Moran_thr_base, M_dyn * 0.9)
        
        # 4. Generate mask
        cond_F = F_abs > F_final
        cond_M = moran_local > M_final
        cond_Valid = ~np.isnan(F_abs) & ~np.isinf(F_abs) & ~np.isnan(moran_local)
        
        mask = cond_F & cond_M & cond_Valid & ctx.inROI
        mask = mask.astype(float)
        
        # 5. Morphological denoising (both.m Line 473)
        # Using opening operation with 3x3 square structuring element
        se = generate_binary_structure(2, 1)  # 3x3 connectivity
        mask = grey_opening(mask, footprint=se)
        
        # Return results
        result = {
            'mask': mask,
            'debug': {
                'F_abs': F_abs,
                'moran_local': moran_local
            }
        }
        
        return result
