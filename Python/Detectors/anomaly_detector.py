"""
AnomalyDetector - Abstract base class for anomaly detectors
Converted from MATLAB AnomalyDetector.m
"""

from abc import ABC, abstractmethod


class AnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detectors
    All detectors must implement the calculate method
    """
    
    @abstractmethod
    def calculate(self, dataCtx):
        """
        Calculate anomaly detection
        
        Parameters:
        -----------
        dataCtx : GeoDataContext
            The geographic data context
        
        Returns:
        --------
        result : dict
            Dictionary with keys:
            - 'mask': binary matrix (anomaly mask)
            - 'debug': dict with intermediate results
        """
        pass
