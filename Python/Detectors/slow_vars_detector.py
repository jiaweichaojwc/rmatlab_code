"""
SlowVarsDetector - Slow variables detector
Converted from MATLAB SlowVarsDetector.m
"""

import numpy as np
from skimage import feature
from skimage.morphology import remove_small_objects, dilation, disk, binary_dilation
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Detectors.anomaly_detector import AnomalyDetector


class SlowVarsDetector(AnomalyDetector):
    """
    Slow variables sudden change detector
    Detects anomalies based on geological slow variable changes
    """
    
    def calculate(self, ctx):
        """Calculate slow variable anomalies"""
        inROI = ctx.inROI
        
        # 1. Geostress
        gy, gx = np.gradient(ctx.dem)
        stress_grad = np.sqrt(gx**2 + gy**2)
        
        # 2. Redox
        iron_oxide = ctx.lan[:, :, 2] / (ctx.lan[:, :, 1] + np.finfo(float).eps)  # B3/B2 (0-indexed: 2/1)
        swir1 = ctx.s2[:, :, 7]  # B11 (0-indexed: 7)
        swir2 = ctx.s2[:, :, 8]  # B12 (0-indexed: 8)
        oxy_fug = swir2 - np.mean(np.stack([swir1, swir2], axis=2), axis=2)
        oxy_fug[np.isnan(oxy_fug) | np.isinf(oxy_fug)] = 0
        
        redox_grad = (np.abs(iron_oxide - np.nanmean(iron_oxide[inROI])) + 
                     np.abs(oxy_fug - np.nanmean(oxy_fug[inROI])))
        
        # 3. Fluid overpressure
        tir_mean = np.mean(ctx.ast[:, :, 9:14], axis=2)  # B10-B14 (0-indexed: 9-13)
        ndvi = (ctx.NIR - ctx.Red) / (ctx.NIR + ctx.Red + np.finfo(float).eps)
        fluid_over = tir_mean + 3 * (1 - ndvi)
        
        # 4. Fault activity
        edges = feature.canny(stress_grad, sigma=1.0, low_threshold=0.05, high_threshold=0.25)
        # Remove small connected components
        edges_cleaned = remove_small_objects(edges, min_size=50)
        fault_activity = edges_cleaned.astype(float) * stress_grad
        
        # 5. Cap layer
        carbonate = (ctx.ast[:, :, 5] + ctx.ast[:, :, 7]) / (ctx.ast[:, :, 6] + np.finfo(float).eps)
        
        # 6. Temperature gradient
        gty, gtx = np.gradient(tir_mean)
        temp_grad = np.sqrt(gtx**2 + gty**2)
        
        # 7. Chemical potential
        gcy, gcx = np.gradient(iron_oxide + oxy_fug)
        chem_grad = np.sqrt(gcx**2 + gcy**2)
        
        # Z-Score helper function
        def z_score(x):
            x_roi = x[inROI]
            mean_val = np.nanmean(x_roi)
            std_val = np.nanstd(x_roi)
            if std_val == 0:
                std_val = np.finfo(float).eps
            return (x - mean_val) / std_val
        
        # Combine variables
        a = -(0.5 * z_score(carbonate) + 0.5 * z_score(temp_grad))
        b = (0.25 * z_score(stress_grad) + 
             0.2 * z_score(redox_grad) + 
             0.25 * z_score(fluid_over) + 
             0.15 * z_score(fault_activity) + 
             0.15 * z_score(chem_grad))
        
        Delta = b**2 + (8/27) * a**3
        
        # Generate mask
        mask = (Delta < 0) & inROI
        
        # Remove small objects
        mask_cleaned = remove_small_objects(mask, min_size=100)
        
        # Dilate
        mask_dilated = binary_dilation(mask_cleaned, disk(8))
        
        result = {
            'mask': mask_dilated.astype(float),
            'debug': {
                'Delta': Delta
            }
        }
        
        return result
