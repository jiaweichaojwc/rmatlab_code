"""
Fusion Engine for managing and combining multiple anomaly detectors.
"""

from typing import Dict, List, Any
import numpy as np
from scipy.ndimage import zoom


class FusionEngine:
    """
    Central orchestrator for multiple anomaly detection algorithms.
    
    Manages detector registration, parallel computation, and mask fusion.
    """
    
    def __init__(self):
        """Initialize the fusion engine with empty detector and result registries."""
        self.detectors: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def add_detector(self, name: str, detector_obj: Any) -> None:
        """
        Register a detector with a unique name.
        
        Args:
            name: Unique identifier for the detector
            detector_obj: Instance of a class implementing AnomalyDetector interface
        """
        self.detectors[name] = detector_obj
    
    def compute_all(self, data_ctx: Any) -> None:
        """
        Execute all registered detectors on the provided data context.
        
        Args:
            data_ctx: GeoDataContext object with loaded geospatial data
        """
        for name, detector in self.detectors.items():
            print(f"  -> Computing detector: {name} ...")
            # Call the calculate method of each detector
            self.results[name] = detector.calculate(data_ctx)
    
    def get_fused_mask(self, names_list: List[str]) -> np.ndarray:
        """
        Fuse multiple detector masks using OR logic.
        
        Args:
            names_list: List of detector names to fuse
            
        Returns:
            Fused binary mask as numpy array
            
        Raises:
            ValueError: If a requested detector hasn't been computed
        """
        if isinstance(names_list, str):
            names_list = [names_list]
        
        fused_mask = None
        
        for name in names_list:
            if name not in self.results:
                raise ValueError(f"Detector {name} has not been computed or registered")
            
            res = self.results[name]
            current_mask = res['mask']
            
            # Ensure mask is float type for consistent operations
            if current_mask.dtype != np.float32 and current_mask.dtype != np.float64:
                current_mask = current_mask.astype(float)
            
            # Initialize or resize and fuse
            if fused_mask is None:
                fused_mask = current_mask.copy()
            else:
                # Ensure same dimensions
                if fused_mask.shape != current_mask.shape:
                    # Resize current mask to match fused mask
                    zoom_factors = (fused_mask.shape[0] / current_mask.shape[0],
                                  fused_mask.shape[1] / current_mask.shape[1])
                    current_mask = zoom(current_mask, zoom_factors, order=0)
                
                # OR logic fusion (using np.logical_or for type safety)
                fused_mask = np.logical_or(fused_mask, current_mask).astype(float)
        
        fused_mask = fused_mask.astype(float)
        print(f">>> Fusion complete, includes: {', '.join(names_list)}")
        
        return fused_mask
    
    def get_result(self, name: str) -> Dict[str, Any]:
        """
        Retrieve results for a specific detector.
        
        Args:
            name: Detector name
            
        Returns:
            Dictionary containing 'mask' and 'debug' data
        """
        return self.results[name]
