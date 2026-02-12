"""
Intrinsic Detector for anomaly detection based on ASTER intrinsic absorption.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import opening, square
from typing import Dict, Any
from .anomaly_detector import AnomalyDetector
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.geo_utils import GeoUtils


class IntrinsicDetector(AnomalyDetector):
    """
    Detects anomalies using intrinsic absorption from ASTER thermal IR data.
    
    Intrinsic absorption relates to mineral composition and can indicate
    the presence of ore deposits.
    """
    
    def calculate(self, ctx) -> Dict[str, Any]:
        """
        Calculate intrinsic absorption anomaly mask.
        
        Args:
            ctx: GeoDataContext object with loaded ASTER data
            
        Returns:
            Dictionary with 'mask' (binary array) and 'debug' (intermediate results)
        """
        # 1. Calculate raw intrinsic absorption
        F_abs_raw = GeoUtils.compute_intrinsic_absorption(ctx.ast, ctx.mineral_type)
        
        # Normalize to 0-1 within ROI
        F_abs = GeoUtils.mat2gray_roi(F_abs_raw, ctx.in_roi)
        
        # 2. Calculate Moran I
        
        # (A) Z-score normalization (strictly within ROI)
        F_vals = F_abs[ctx.in_roi]
        F_mean = np.nanmean(F_vals)
        F_std = np.nanstd(F_vals)
        
        if F_std == 0:
            F_std = np.finfo(float).eps
        
        Z = (F_abs - F_mean) / F_std
        Z[~ctx.in_roi] = np.nan
        
        # (B) Calculate local sum with NaN handling
        local_sum = GeoUtils.calc_local_sum_with_nan(Z)
        
        # (C) Normalize local convolution sum
        ls_roi = local_sum[ctx.in_roi]
        max_ls = np.nanmax(ls_roi)
        
        if max_ls == 0 or np.isnan(max_ls):
            max_ls = np.finfo(float).eps
        
        moran_local = Z * local_sum / max_ls
        
        # (D) Clean up invalid values
        moran_local[~ctx.in_roi] = np.nan
        moran_local[np.isnan(moran_local) | np.isinf(moran_local)] = 0
        
        # 3. Dynamic threshold calculation (95th percentile)
        F_thr_base, _, Moran_thr_base, _ = GeoUtils.get_mineral_thresholds(ctx.mineral_type)
        
        # Levashov mode check
        if getattr(ctx, 'levashov_mode', False):
            F_thr_base = F_thr_base * 0.8
            Moran_thr_base = Moran_thr_base * 0.8
        
        # Extract valid ROI values for statistics
        F_roi_vals = F_abs[ctx.in_roi & ~np.isnan(F_abs)]
        m_roi_vals = moran_local[ctx.in_roi & ~np.isnan(moran_local)]
        
        if len(F_roi_vals) > 0:
            F_dyn = np.percentile(F_roi_vals, 95)
        else:
            F_dyn = 0
        
        if len(m_roi_vals) > 0:
            M_dyn = np.percentile(m_roi_vals, 95)
        else:
            M_dyn = 0
        
        # Fuse thresholds
        F_final = max(F_thr_base, F_dyn * 0.9)
        M_final = max(Moran_thr_base, M_dyn * 0.9)
        
        # 4. Generate mask
        cond_F = F_abs > F_final
        cond_M = moran_local > M_final
        cond_Valid = ~np.isnan(F_abs) & ~np.isinf(F_abs) & ~np.isnan(moran_local)
        
        mask = cond_F & cond_M & cond_Valid & ctx.in_roi
        mask = mask.astype(float)
        
        # 5. Morphological denoising
        selem = square(3)
        mask = opening(mask, selem)
        mask = mask.astype(float)  # Convert back to float after morphological operation
        
        # Return results
        result = {
            'mask': mask,
            'debug': {
                'F_abs': F_abs,
                'moran_local': moran_local
            }
        }
        
        return result
