"""
Known Anomaly Detector using KML/KMZ files.
Extracts anomaly masks from KML/KMZ geographic data aligned with reference imagery.
"""
import os
from typing import Dict, Any
import numpy as np
from scipy.ndimage import zoom

from detectors.anomaly_detector import AnomalyDetector
from utils.kmz_mask_generator import KMZMaskGenerator


class KnownAnomalyDetector(AnomalyDetector):
    """
    Detector that extracts known anomalies from KML/KMZ files.
    
    Uses KMZMaskGenerator to process geographic data and align with image ROI.
    """
    
    def calculate(self, ctx) -> Dict[str, Any]:
        """
        Calculate known anomaly mask from KML/KMZ file.
        
        Args:
            ctx: GeoDataContext object with required attributes:
                - kmz_path: Path to KML/KMZ file
                - ref_tif_path: Path to reference GeoTIFF
                - kmz_keywords: List of keywords to match (optional)
                - inROI: Boolean mask defining region of interest
                
        Returns:
            Dictionary with:
                - mask: Binary anomaly mask aligned with ROI
                - debug: Dictionary with raw mask data
        """
        print('  [KnownAnomaly] 正在处理 KML 已知异常...')
        
        # Initialize result structure
        res = {
            'mask': np.zeros(ctx.inROI.shape, dtype=float),
            'debug': {'raw': None}
        }
        
        # 1. Check configuration
        if not hasattr(ctx, 'kmz_path') or not ctx.kmz_path or not os.path.exists(ctx.kmz_path):
            print('    ⚠️ KML文件不存在或未选择，跳过。')
            return res
        
        if not hasattr(ctx, 'ref_tif_path') or not ctx.ref_tif_path:
            print('    ❌ 缺少参考影像路径(ref_tif_path)，无法校正KML坐标。')
            return res
        
        # 2. Call KMZMaskGenerator
        try:
            radius = 3  # Default expansion radius
            
            # Get keywords from context or use defaults
            keywords = getattr(ctx, 'kmz_keywords', None)
            
            # Instantiate generator
            generator = KMZMaskGenerator(
                ctx.kmz_path,
                ctx.ref_tif_path,
                keywords,
                radius
            )
            
            # Run generation
            raw_mask = generator.run()
            
            # 3. Size alignment
            target_size = ctx.inROI.shape
            if raw_mask.shape != target_size:
                print(f'    ⚠️ 尺寸不匹配 ({raw_mask.shape[0]}x{raw_mask.shape[1]} vs '
                      f'{target_size[0]}x{target_size[1]})，调整中...')
                raw_mask = self._resize_mask(raw_mask, target_size)
            
            # 4. Result packaging
            mask = raw_mask.astype(float)
            mask[~ctx.inROI] = 0  # Clip to ROI
            
            res['mask'] = mask
            res['debug']['raw'] = raw_mask
            
            print(f'    ✅ KML异常提取完成，像素数: {int(np.sum(mask > 0))}')
            
        except Exception as e:
            print(f'    ❌ KML处理出错: {str(e)}')
            res['mask'] = np.zeros(ctx.inROI.shape, dtype=float)
        
        return res
    
    def _resize_mask(self, mask: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Resize mask to target size using nearest neighbor interpolation.
        
        Args:
            mask: Input binary mask
            target_size: Target (height, width)
            
        Returns:
            Resized mask
        """
        # Calculate zoom factors
        zoom_factors = (
            target_size[0] / mask.shape[0],
            target_size[1] / mask.shape[1]
        )
        
        # Use nearest neighbor (order=0) to preserve binary nature
        resized = zoom(mask.astype(float), zoom_factors, order=0)
        
        return resized.astype(bool)
