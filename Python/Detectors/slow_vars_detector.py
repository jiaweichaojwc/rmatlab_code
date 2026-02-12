"""
Slow Variables Detector using Cardano's discriminant for stability analysis.
"""

import numpy as np
from scipy.ndimage import generic_filter
from skimage.filters import canny
from skimage.morphology import binary_opening, remove_small_objects, binary_dilation, disk
from typing import Dict, Any
from .anomaly_detector import AnomalyDetector


class SlowVarsDetector(AnomalyDetector):
    """
    Detects anomalies using slow-varying geophysical/geochemical parameters.
    
    Uses Cardano's discriminant on a Gibbs free energy proxy to identify
    thermodynamic instability zones (potential mineralization sites).
    """
    
    def calculate(self, ctx) -> Dict[str, Any]:
        """
        Calculate slow variables anomaly mask using multiple physical parameters.
        
        Args:
            ctx: GeoDataContext object
            
        Returns:
            Dictionary with 'mask' (binary array) and 'debug' (intermediate results)
        """
        in_roi = ctx.in_roi
        eps = np.finfo(float).eps
        
        # 1. Stress gradient (from DEM)
        gx, gy = np.gradient(ctx.dem)
        stress_grad = np.sqrt(gx**2 + gy**2)
        
        # 2. Redox gradient
        iron_oxide = ctx.lan[:, :, 2] / (ctx.lan[:, :, 1] + eps)
        swir1 = ctx.s2[:, :, 7]
        swir2 = ctx.s2[:, :, 8]
        oxy_fug = swir2 - np.mean(np.stack([swir1, swir2], axis=-1), axis=-1)
        oxy_fug[np.isnan(oxy_fug) | np.isinf(oxy_fug)] = 0
        
        redox_grad = (np.abs(iron_oxide - np.nanmean(iron_oxide[in_roi])) + 
                      np.abs(oxy_fug - np.nanmean(oxy_fug[in_roi])))
        
        # 3. Fluid overpressure
        tir_mean = np.mean(ctx.ast[:, :, 9:14], axis=2)  # ASTER bands 10-14 (0-indexed)
        ndvi = (ctx.nir - ctx.red) / (ctx.nir + ctx.red + eps)
        fluid_over = tir_mean + 3 * (1 - ndvi)
        
        # 4. Fault activity
        edges = canny(stress_grad, sigma=1.0, low_threshold=0.05, high_threshold=0.25)
        # Remove small objects (equivalent to bwareaopen)
        fault_activity = remove_small_objects(edges, min_size=50).astype(float) * stress_grad
        
        # 5. Carbonate cap layer
        carbonate = (ctx.ast[:, :, 5] + ctx.ast[:, :, 7]) / (ctx.ast[:, :, 6] + eps)
        
        # 6. Temperature gradient
        gtx, gty = np.gradient(tir_mean)
        temp_grad = np.sqrt(gtx**2 + gty**2)
        
        # 7. Chemical potential gradient
        gcx, gcy = np.gradient(iron_oxide + oxy_fug)
        chem_grad = np.sqrt(gcx**2 + gcy**2)
        
        # Z-score normalization helper
        def z_score(x):
            vals = x[in_roi]
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals)
            if std_val == 0:
                std_val = eps
            return (x - mean_val) / std_val
        
        # Combine variables using weighted Cardano discriminant
        a = -(0.5 * z_score(carbonate) + 0.5 * z_score(temp_grad))
        b = (0.25 * z_score(stress_grad) + 
             0.2 * z_score(redox_grad) + 
             0.25 * z_score(fluid_over) + 
             0.15 * z_score(fault_activity) + 
             0.15 * z_score(chem_grad))
        
        # Cardano's discriminant: Δ = b² + (8/27)a³
        # Negative discriminant indicates complex roots (unstable phase, potential mineralization)
        Delta = b**2 + (8/27) * a**3
        mask = (Delta < 0) & in_roi
        
        # Morphological cleanup
        mask = remove_small_objects(mask, min_size=100)
        mask = binary_dilation(mask, disk(8))
        
        result = {
            'mask': mask.astype(float),
            'debug': {
                'Delta': Delta
            }
        }
        
        return result
