"""
Fusion Engine for combining multiple detector results.
Equivalent to FusionEngine.m
"""
from typing import Dict, List, Union
import numpy as np
from scipy.ndimage import zoom


class FusionEngine:
    """
    Engine for managing detectors and fusing their results.
    """
    
    def __init__(self):
        """Initialize the fusion engine."""
        self.detectors: Dict[str, any] = {}  # Name -> DetectorObject
        self.results: Dict[str, Dict] = {}   # Name -> ResultDict
        
    def add_detector(self, name: str, detector_obj) -> None:
        """
        Register a detector with a name.
        
        Args:
            name: Name identifier for the detector
            detector_obj: Instance of AnomalyDetector subclass
        """
        self.detectors[name] = detector_obj
        
    def compute_all(self, data_ctx) -> None:
        """
        Execute all registered detectors.
        
        Args:
            data_ctx: GeoDataContext object containing data
        """
        for name, detector in self.detectors.items():
            print(f'  -> Computing detector: {name} ...')
            # Call the calculate method of each detector
            self.results[name] = detector.calculate(data_ctx)
            
    def get_fused_mask(self, names_list: Union[str, List[str]]) -> np.ndarray:
        """
        Fuse multiple detector masks using OR logic.
        
        Args:
            names_list: Single name or list of detector names to fuse
            
        Returns:
            Fused binary mask as numpy array
        """
        if isinstance(names_list, str):
            names_list = [names_list]
            
        fused_mask = None
        
        for name in names_list:
            if name not in self.results:
                raise ValueError(f'Detector {name} has not been computed or registered')
                
            res = self.results[name]
            current_mask = res['mask']
            
            # Initialize or resize to match
            if fused_mask is None:
                fused_mask = current_mask.copy()
            else:
                if fused_mask.shape != current_mask.shape:
                    # Resize current_mask to match fused_mask using nearest neighbor
                    zoom_factors = (fused_mask.shape[0] / current_mask.shape[0],
                                    fused_mask.shape[1] / current_mask.shape[1])
                    current_mask = zoom(current_mask, zoom_factors, order=0)
                    
                # OR logic fusion
                fused_mask = np.logical_or(fused_mask, current_mask)
                
        fused_mask = fused_mask.astype(float)
        print(f'>>> Fusion complete, including: {", ".join(names_list)}')
        return fused_mask
        
    def get_result(self, name: str) -> Dict:
        """
        Get result for a specific detector.
        
        Args:
            name: Detector name
            
        Returns:
            Result dictionary with 'mask' and 'debug' keys
        """
        return self.results[name]
