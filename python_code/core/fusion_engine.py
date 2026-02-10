"""
Fusion Engine for combining multiple anomaly detection results.
融合引擎 - 用于组合多个异常探测结果
"""

from typing import Dict, List, Union
import numpy as np
from skimage.transform import resize


class FusionEngine:
    """
    Engine for managing multiple anomaly detectors and fusing their results.
    """
    
    def __init__(self):
        """Initialize the fusion engine with empty detector and result maps."""
        self.detectors: Dict[str, any] = {}
        self.results: Dict[str, Dict] = {}
    
    def add_detector(self, name: str, detector_obj):
        """
        Register a detector with a given name.
        
        Parameters:
        -----------
        name : str
            Name identifier for the detector
        detector_obj : AnomalyDetector
            Instance of an anomaly detector
        """
        self.detectors[name] = detector_obj
    
    def compute_all(self, data_ctx):
        """
        Run all registered detectors on the provided data context.
        
        Parameters:
        -----------
        data_ctx : GeoDataContext
            The geographical data context containing all input data
        """
        for name, detector in self.detectors.items():
            print(f"  -> 计算探测器: {name} ...")
            self.results[name] = detector.calculate(data_ctx)
    
    def get_fused_mask(self, names_list: Union[str, List[str]]) -> np.ndarray:
        """
        Fuse masks from specified detectors using OR logic.
        
        Parameters:
        -----------
        names_list : str or list of str
            Name(s) of detector(s) to fuse
            
        Returns:
        --------
        numpy.ndarray
            Fused binary mask
        """
        if isinstance(names_list, str):
            names_list = [names_list]
        
        fused_mask = None
        
        for name in names_list:
            if name not in self.results:
                raise KeyError(f'探测器 {name} 尚未计算或未注册')
            
            res = self.results[name]
            current_mask = res['mask']
            
            # Initialize or resize to match dimensions
            if fused_mask is None:
                fused_mask = current_mask.copy()
            else:
                if fused_mask.shape != current_mask.shape:
                    # Resize using nearest neighbor to maintain binary values
                    current_mask = resize(
                        current_mask, 
                        fused_mask.shape, 
                        order=0,  # nearest neighbor
                        preserve_range=True,
                        anti_aliasing=False
                    )
                # OR logic fusion
                fused_mask = np.logical_or(fused_mask, current_mask)
        
        fused_mask = fused_mask.astype(float)
        print(f'>>> 融合完成，包含了: {", ".join(names_list)}')
        return fused_mask
    
    def get_result(self, name: str) -> Dict:
        """
        Get the result of a specific detector.
        
        Parameters:
        -----------
        name : str
            Name of the detector
            
        Returns:
        --------
        dict
            Result dictionary containing mask and debug information
        """
        return self.results[name]
