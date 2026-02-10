"""
Slow Variables Detector for geophysical anomaly detection.
慢变量探测器 - 用于地球物理异常探测
"""

import numpy as np
from scipy.ndimage import binary_opening, binary_dilation
from skimage.filters import canny
from skimage.morphology import remove_small_objects, disk
from .anomaly_detector import AnomalyDetector


class SlowVarsDetector(AnomalyDetector):
    """
    Detector based on slow-varying geophysical parameters.
    基于慢变地球物理参数的探测器
    """
    
    def calculate(self, ctx) -> dict:
        """
        Calculate slow variable anomalies.
        
        Parameters:
        -----------
        ctx : GeoDataContext
            Data context containing all input data
            
        Returns:
        --------
        dict
            Result dictionary with 'mask' and 'debug' keys
        """
        inROI = ctx.inROI
        
        # 1. Geostress
        gy, gx = np.gradient(ctx.dem)
        stress_grad = np.sqrt(gx**2 + gy**2)
        
        # 2. Redox
        iron_oxide = ctx.lan[:, :, 2] / (ctx.lan[:, :, 1] + np.finfo(float).eps)
        swir1 = ctx.s2[:, :, 7]
        swir2 = ctx.s2[:, :, 8]
        oxy_fug = swir2 - np.mean(np.stack([swir1, swir2], axis=-1), axis=-1)
        oxy_fug[np.isnan(oxy_fug) | np.isinf(oxy_fug)] = 0
        
        redox_grad = (np.abs(iron_oxide - np.nanmean(iron_oxide[inROI])) + 
                     np.abs(oxy_fug - np.nanmean(oxy_fug[inROI])))
        
        # 3. Fluid overpressure
        tir_mean = np.mean(ctx.ast[:, :, 9:14], axis=-1)
        ndvi = (ctx.NIR - ctx.Red) / (ctx.NIR + ctx.Red + np.finfo(float).eps)
        fluid_over = tir_mean + 3 * (1 - ndvi)
        
        # 4. Fault activity
        edges = canny(stress_grad, sigma=1.0, low_threshold=0.05, high_threshold=0.25)
        # Remove small objects (< 50 pixels)
        edges_cleaned = remove_small_objects(edges, min_size=50)
        fault_activity = edges_cleaned.astype(float) * stress_grad
        
        # 5. Cap layer (carbonate)
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
        
        # Discriminant
        Delta = b**2 + (8/27) * a**3
        
        # Generate mask
        mask = (Delta < 0) & inROI
        
        # Morphological processing
        mask = remove_small_objects(mask, min_size=100)
        mask = binary_dilation(mask, structure=disk(8))
        
        res = {
            'mask': mask.astype(float),
            'debug': {'Delta': Delta}
        }
        
        return res
