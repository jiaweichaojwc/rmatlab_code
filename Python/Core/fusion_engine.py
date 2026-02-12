"""
FusionEngine - Detector fusion engine
Converted from MATLAB FusionEngine.m
"""

import numpy as np
from scipy.ndimage import zoom


class FusionEngine:
    """
    Fusion engine for combining multiple detector results
    """
    
    def __init__(self):
        self.detectors = {}  # Dict: Name -> DetectorObject
        self.results = {}    # Dict: Name -> ResultStruct (mask, debug_data)
    
    def addDetector(self, name, detector_obj):
        """Add a detector to the engine"""
        self.detectors[name] = detector_obj
    
    def computeAll(self, dataCtx):
        """Compute all registered detectors"""
        for name, detector in self.detectors.items():
            print(f'  -> 计算探测器: {name} ...')
            # Call specific detector's calculate method
            self.results[name] = detector.calculate(dataCtx)
    
    def getFusedMask(self, names_list):
        """
        Flexible fusion: pass ['RedEdge', 'Intrinsic'] to fuse only these two
        
        Parameters:
        -----------
        names_list : list or str
            List of detector names to fuse
        
        Returns:
        --------
        fused_mask : ndarray
            Fused binary mask
        """
        if isinstance(names_list, str):
            names_list = [names_list]
        
        fused_mask = None
        
        for name in names_list:
            if name not in self.results:
                raise ValueError(f'探测器 {name} 尚未计算或未注册')
            
            res = self.results[name]
            current_mask = res['mask']
            
            # Ensure mask is boolean or numeric
            if not isinstance(current_mask, np.ndarray):
                current_mask = np.array(current_mask)
            
            # Unify size
            if fused_mask is None:
                fused_mask = current_mask.copy()
            else:
                if fused_mask.shape != current_mask.shape:
                    # Resize current mask to match fused mask
                    zoom_factors = (fused_mask.shape[0] / current_mask.shape[0],
                                   fused_mask.shape[1] / current_mask.shape[1])
                    current_mask = zoom(current_mask, zoom_factors, order=0)
                
                # Convert to boolean for OR operation, then back to float
                fused_mask_bool = fused_mask.astype(bool)
                current_mask_bool = current_mask.astype(bool)
                fused_mask = (fused_mask_bool | current_mask_bool).astype(float)
        
        fused_mask = fused_mask.astype(float)
        print(f'>>> 融合完成，包含了: {", ".join(names_list)}')
        
        return fused_mask
    
    def getResult(self, name):
        """Get result for a specific detector"""
        return self.results[name]
