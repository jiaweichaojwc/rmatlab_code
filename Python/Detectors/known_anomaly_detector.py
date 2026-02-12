"""
Known Anomaly Detector for extracting anomalies from KML/KMZ files.
"""

import numpy as np
from typing import Dict, Any
from .anomaly_detector import AnomalyDetector
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.kmz_mask_generator import KMZMaskGenerator


class KnownAnomalyDetector(AnomalyDetector):
    """
    Detects known anomalies from KML/KMZ files.
    
    This detector rasterizes KML/KMZ geometry data to create a binary mask
    of known mineral deposit locations or anomalies.
    """
    
    def calculate(self, ctx) -> Dict[str, Any]:
        """
        Extract known anomalies from KML/KMZ file.
        
        Args:
            ctx: GeoDataContext object with kmz_path configuration
            
        Returns:
            Dictionary with 'mask' (binary array) and 'debug' (raw mask)
        """
        print("  [KnownAnomaly] Processing KML known anomalies...")
        
        # 1. Check configuration
        if not hasattr(ctx, 'kmz_path') or not ctx.kmz_path or not os.path.exists(ctx.kmz_path):
            print("    ⚠️ KML file does not exist or not selected, skipping.")
            return {
                'mask': np.zeros_like(ctx.in_roi, dtype=float),
                'debug': {'raw': None}
            }
        
        if not hasattr(ctx, 'ref_tif_path') or not ctx.ref_tif_path:
            print("    ❌ Missing reference image path (ref_tif_path), cannot rectify KML coordinates.")
            return {
                'mask': np.zeros_like(ctx.in_roi, dtype=float),
                'debug': {'raw': None}
            }
        
        # 2. Call KMZMaskGenerator
        try:
            radius = 3  # Default expansion radius for point features
            
            # Get keywords
            keywords = getattr(ctx, 'kmz_keywords', ['矿体投影', 'Object ID', 'ZK', '异常', '已知矿点'])
            
            # Instantiate generator
            generator = KMZMaskGenerator(ctx.kmz_path, ctx.ref_tif_path, keywords, radius)
            
            # Run generation
            raw_mask = generator.run()
            
            # 3. Size alignment
            target_size = ctx.in_roi.shape
            if raw_mask.shape != target_size:
                print(f"    ⚠️ Size mismatch ({raw_mask.shape} vs {target_size}), resizing...")
                from scipy.ndimage import zoom
                zoom_factors = (target_size[0] / raw_mask.shape[0], 
                              target_size[1] / raw_mask.shape[1])
                raw_mask = zoom(raw_mask, zoom_factors, order=0)  # Nearest neighbor
            
            # 4. Result packaging
            mask = raw_mask.astype(float)
            mask[~ctx.in_roi] = 0  # Clip to ROI
            
            print(f"    ✅ KML anomaly extraction complete, pixel count: {np.sum(mask > 0)}")
            
            return {
                'mask': mask,
                'debug': {'raw': raw_mask}
            }
            
        except Exception as e:
            print(f"    ❌ KML processing error: {str(e)}")
            return {
                'mask': np.zeros_like(ctx.in_roi, dtype=float),
                'debug': {'raw': None}
            }
