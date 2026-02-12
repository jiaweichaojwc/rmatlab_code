"""
Abstract base class for anomaly detectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class AnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detection algorithms.
    
    All detector implementations must override the calculate method.
    """
    
    @abstractmethod
    def calculate(self, data_ctx) -> Dict[str, Any]:
        """
        Calculate anomaly detection mask and debug data.
        
        Args:
            data_ctx: GeoDataContext object containing all geospatial data
            
        Returns:
            Dictionary with keys:
                - 'mask': Binary numpy array indicating anomalies (1) vs background (0)
                - 'debug': Dictionary of intermediate results for visualization
        """
        pass
