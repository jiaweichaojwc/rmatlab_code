"""
Intrinsic Absorption Detector for mineral anomaly detection.
本征吸收探测器 - 用于矿产异常探测
"""

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import binary_opening
from scipy.ndimage import generate_binary_structure
from .anomaly_detector import AnomalyDetector
from ..utils.geo_utils import GeoUtils


class IntrinsicDetector(AnomalyDetector):
    """
    Detector based on intrinsic absorption analysis.
    基于本征吸收分析的探测器
    """
    
    def calculate(self, ctx) -> dict:
        """
        Calculate intrinsic absorption anomalies.
        
        Parameters:
        -----------
        ctx : GeoDataContext
            Data context containing all input data
            
        Returns:
        --------
        dict
            Result dictionary with 'mask' and 'debug' keys
        """
        # 1. Calculate raw intrinsic absorption intensity
        F_abs_raw = GeoUtils.compute_intrinsic_absorption(ctx.ast, ctx.mineral_type)
        
        # Normalize to [0, 1] within ROI first (critical step)
        F_abs = GeoUtils.mat2gray_roi(F_abs_raw, ctx.inROI)
        
        # 2. Calculate Moran I (using conv2 approach to match MATLAB)
        # (A) Z-Score standardization
        F_vals = F_abs[ctx.inROI]
        F_mean = np.nanmean(F_vals)
        F_std = np.nanstd(F_vals)
        
        if F_std == 0:
            F_std = np.finfo(float).eps
        
        Z = (F_abs - F_mean) / F_std
        Z[~ctx.inROI] = np.nan
        
        # (B) NaN handling - convert NaN to 0 for convolution
        Z_conv = Z.copy()
        Z_conv[np.isnan(Z_conv) | np.isinf(Z_conv)] = 0
        
        # (C) Convolution for local sum (3x3 kernel, center = 0)
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        local_sum = convolve(Z_conv, kernel, mode='constant', cval=0.0)
        local_sum[np.isnan(local_sum)] = 0
        
        # (D) Normalize by max local sum (not mat2gray to [0,1])
        ls_roi = local_sum[ctx.inROI]
        max_ls = np.nanmax(ls_roi)
        
        if max_ls == 0 or np.isnan(max_ls):
            max_ls = np.finfo(float).eps
        
        moran_local = Z * local_sum / max_ls
        
        # (E) Clean invalid values
        moran_local[~ctx.inROI] = np.nan
        moran_local[np.isnan(moran_local) | np.isinf(moran_local)] = 0
        
        # 3. Dynamic threshold calculation (95th percentile)
        F_thr_base, _, Moran_thr_base, _ = GeoUtils.get_mineral_thresholds(ctx.mineral_type)
        
        # Levashov mode: reduce thresholds by 20%
        if getattr(ctx, 'levashov_mode', False):
            F_thr_base = F_thr_base * 0.8
            Moran_thr_base = Moran_thr_base * 0.8
        
        # Extract ROI valid values for statistics
        F_roi_vals = F_abs[ctx.inROI & ~np.isnan(F_abs)]
        m_roi_vals = moran_local[ctx.inROI & ~np.isnan(moran_local)]
        
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
        
        mask = cond_F & cond_M & cond_Valid & ctx.inROI
        mask = mask.astype(float)
        
        # 5. Morphological noise reduction (opening with 3x3 square)
        struct = generate_binary_structure(2, 1)  # 4-connectivity
        struct = np.ones((3, 3))  # Square structuring element
        mask = binary_opening(mask, structure=struct).astype(float)
        
        # Return results
        res = {
            'mask': mask,
            'debug': {
                'F_abs': F_abs,
                'moran_local': moran_local
            }
        }
        
        return res
