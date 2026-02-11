"""
Abstract base class for anomaly detectors.
Equivalent to AnomalyDetector.m
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class AnomalyDetector(ABC):
    """
    Abstract base class for anomaly detectors.
    All detectors must return: {'mask': binary_matrix, 'debug': {...}}
    """
    
    @abstractmethod
    def calculate(self, data_ctx) -> Dict[str, Any]:
        """
        Calculate anomaly detection result.
        
        Args:
            data_ctx: GeoDataContext object containing all necessary data
            
        Returns:
            Dictionary with 'mask' (binary numpy array) and 'debug' (dict) keys
        """
        pass
