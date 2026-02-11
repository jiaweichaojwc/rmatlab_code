"""
KnownAnomalyDetector - Known anomaly detector from KML/KMZ
Converted from MATLAB KnownAnomalyDetector.m
"""

import numpy as np
from scipy.ndimage import zoom
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Detectors.anomaly_detector import AnomalyDetector
from Utils.kmz_mask_generator import KMZMaskGenerator


class KnownAnomalyDetector(AnomalyDetector):
    """
    Known anomaly detector from KML/KMZ files
    Extracts known mineral deposits from Google Earth files
    """
    
    def calculate(self, ctx):
        """Extract known anomalies from KML"""
        print('  [KnownAnomaly] 正在处理 KML 已知异常...')
        
        # 1. Check configuration
        if not ctx.kmz_path or not os.path.exists(ctx.kmz_path):
            print('    ⚠️ KML文件不存在或未选择，跳过。')
            result = {
                'mask': np.zeros(ctx.inROI.shape),
                'debug': {'raw': []}
            }
            return result
        
        if not ctx.ref_tif_path:
            print('    ❌ 缺少参考影像路径(ref_tif_path)，无法校正KML坐标。')
            result = {
                'mask': np.zeros(ctx.inROI.shape),
                'debug': {'raw': []}
            }
            return result
        
        # 2. Call KMZMaskGenerator
        try:
            radius = 3  # Default expansion radius
            # Instantiate generator
            generator = KMZMaskGenerator(
                ctx.kmz_path, 
                ctx.ref_tif_path, 
                ctx.kmz_keywords, 
                radius
            )
            
            # Run generation
            raw_mask = generator.run()
            
            # 3. Size alignment
            target_size = ctx.inROI.shape
            if raw_mask.shape != target_size:
                print(f'    ⚠️ 尺寸不匹配 ({raw_mask.shape[0]}x{raw_mask.shape[1]} vs '
                      f'{target_size[0]}x{target_size[1]})，调整中...')
                # Resize using nearest neighbor
                zoom_factors = (target_size[0] / raw_mask.shape[0], 
                               target_size[1] / raw_mask.shape[1])
                raw_mask = zoom(raw_mask.astype(float), zoom_factors, order=0) > 0.5
            
            # 4. Result packaging
            mask = raw_mask.astype(float)
            mask[~ctx.inROI] = 0  # Clip to ROI
            
            result = {
                'mask': mask,
                'debug': {'raw': raw_mask}
            }
            
            print(f'    ✅ KML异常提取完成，像素数: {int(np.sum(mask > 0))}')
            
        except Exception as e:
            print(f'    ❌ KML处理出错: {str(e)}')
            result = {
                'mask': np.zeros(ctx.inROI.shape),
                'debug': {'raw': []}
            }
        
        return result
