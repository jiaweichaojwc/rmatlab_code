"""
Known Anomaly Detector for incorporating KML/KMZ data.
已知异常探测器 - 用于合并KML/KMZ数据
"""

import numpy as np
from .anomaly_detector import AnomalyDetector
from ..utils.kmz_mask_generator import KMZMaskGenerator
from skimage.transform import resize


class KnownAnomalyDetector(AnomalyDetector):
    """
    Detector for incorporating known anomalies from KML/KMZ files.
    用于合并KML/KMZ文件中的已知异常
    """
    
    def calculate(self, ctx) -> dict:
        """
        Process KML/KMZ known anomaly data.
        
        Parameters:
        -----------
        ctx : GeoDataContext
            Data context containing all input data
            
        Returns:
        --------
        dict
            Result dictionary with 'mask' and 'debug' keys
        """
        print("  [KnownAnomaly] 正在处理 KML 已知异常...")
        
        # 1. Check configuration
        if not hasattr(ctx, 'kmz_path') or not ctx.kmz_path or not ctx.kmz_path.strip():
            print("    ⚠️ KML文件不存在或未选择，跳过。")
            return {
                'mask': np.zeros_like(ctx.inROI, dtype=float),
                'debug': {'raw': []}
            }
        
        import os
        if not os.path.exists(ctx.kmz_path):
            print(f"    ⚠️ KML文件不存在: {ctx.kmz_path}")
            return {
                'mask': np.zeros_like(ctx.inROI, dtype=float),
                'debug': {'raw': []}
            }
        
        if not hasattr(ctx, 'ref_tif_path') or not ctx.ref_tif_path:
            print("    ❌ 缺少参考影像路径(ref_tif_path)，无法校正KML坐标。")
            return {
                'mask': np.zeros_like(ctx.inROI, dtype=float),
                'debug': {'raw': []}
            }
        
        # 2. Call KMZMaskGenerator
        try:
            radius = 3  # Default expansion radius
            generator = KMZMaskGenerator(ctx.kmz_path, ctx.ref_tif_path, 
                                        ctx.kmz_keywords, radius)
            
            raw_mask = generator.run()
            
            # 3. Resize to match dimensions
            target_size = ctx.inROI.shape
            if raw_mask.shape != target_size:
                print(f"    ⚠️ 尺寸不匹配 ({raw_mask.shape} vs {target_size})，调整中...")
                raw_mask = resize(raw_mask, target_size, order=0, 
                                preserve_range=True, anti_aliasing=False)
            
            # 4. Package results
            mask = raw_mask.astype(float)
            mask[~ctx.inROI] = 0  # Clip to ROI
            
            print(f"    ✅ KML异常提取完成，像素数: {np.sum(mask > 0)}")
            
            return {
                'mask': mask,
                'debug': {'raw': raw_mask}
            }
            
        except Exception as e:
            print(f"    ❌ KML处理出错: {e}")
            return {
                'mask': np.zeros_like(ctx.inROI, dtype=float),
                'debug': {'raw': []}
            }
