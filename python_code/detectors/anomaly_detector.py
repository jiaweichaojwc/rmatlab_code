"""
Base class for anomaly detectors.
异常探测器基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class AnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detectors.
    All detectors must implement the calculate method.
    """
    
    @abstractmethod
    def calculate(self, data_ctx) -> Dict[str, Any]:
        """
        Calculate anomaly detection results.
        
        Parameters:
        -----------
        data_ctx : GeoDataContext
            The geographical data context containing all input data
            
        Returns:
        --------
        dict
            A dictionary containing:
            - 'mask': Binary mask (numpy array) indicating anomaly locations
            - 'debug': Dictionary with intermediate results for debugging/visualization
        """
        pass
