from .anomaly_detector import AnomalyDetector

# Conditional imports to avoid breaking when run as script
try:
    from .red_edge_detector import RedEdgeDetector
    from .intrinsic_detector import IntrinsicDetector
    from .slow_vars_detector import SlowVarsDetector
    from .known_anomaly_detector import KnownAnomalyDetector
    
    __all__ = [
        'AnomalyDetector',
        'RedEdgeDetector',
        'IntrinsicDetector',
        'SlowVarsDetector',
        'KnownAnomalyDetector'
    ]
except ImportError:
    # When running as script, only export base class
    __all__ = ['AnomalyDetector']
