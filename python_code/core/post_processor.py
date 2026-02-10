"""
Post-processor for results visualization and export.
后处理器 - 用于结果可视化和导出
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class PostProcessor:
    """
    Post-processing for visualization and result export.
    用于可视化和结果导出的后处理
    """
    
    @staticmethod
    def run(data_ctx, engine, final_mask, out_dir: str):
        """
        Run post-processing pipeline.
        
        Parameters:
        -----------
        data_ctx : GeoDataContext
            Data context
        engine : FusionEngine
            Fusion engine with computed results
        final_mask : numpy.ndarray
            Final fused mask
        out_dir : str
            Output directory path
        """
        import os
        os.makedirs(out_dir, exist_ok=True)
        
        print(f">>> 开始后处理与绘图...")
        print(f"    输出目录: {out_dir}")
        
        # 1. Save final mask
        try:
            mask_file = os.path.join(out_dir, 'final_mask.npy')
            np.save(mask_file, final_mask)
            print(f"    ✓ 保存最终掩码: {mask_file}")
        except Exception as e:
            print(f"    警告: 无法保存掩码文件: {e}")
        
        # 2. Create visualization
        try:
            PostProcessor._plot_results(data_ctx, final_mask, out_dir)
        except Exception as e:
            print(f"    警告: 可视化失败: {e}")
        
        # 3. Export statistics
        try:
            PostProcessor._export_statistics(data_ctx, engine, final_mask, out_dir)
        except Exception as e:
            print(f"    警告: 统计导出失败: {e}")
        
        print(">>> 后处理完成")
    
    @staticmethod
    def _plot_results(data_ctx, final_mask, out_dir: str):
        """
        Create visualization plots.
        
        Parameters:
        -----------
        data_ctx : GeoDataContext
            Data context
        final_mask : numpy.ndarray
            Final mask
        out_dir : str
            Output directory
        """
        import os
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: RGB composite (if available)
        if data_ctx.Red is not None and data_ctx.Green is not None and data_ctx.Blue is not None:
            rgb = np.stack([
                PostProcessor._normalize(data_ctx.Red),
                PostProcessor._normalize(data_ctx.Green),
                PostProcessor._normalize(data_ctx.Blue)
            ], axis=-1)
            axes[0].imshow(rgb)
            axes[0].set_title('RGB Composite')
            axes[0].axis('off')
        else:
            axes[0].text(0.5, 0.5, 'RGB数据不可用', ha='center', va='center')
            axes[0].axis('off')
        
        # Plot 2: Anomaly mask
        axes[1].imshow(final_mask, cmap='hot', interpolation='nearest')
        axes[1].set_title('Anomaly Detection Result')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(out_dir, 'results_visualization.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ 保存可视化图像: {fig_path}")
    
    @staticmethod
    def _normalize(band: np.ndarray) -> np.ndarray:
        """
        Normalize band to [0, 1] for visualization.
        
        Parameters:
        -----------
        band : numpy.ndarray
            Input band
            
        Returns:
        --------
        numpy.ndarray
            Normalized band
        """
        valid = band[~np.isnan(band) & ~np.isinf(band)]
        if len(valid) == 0:
            return np.zeros_like(band)
        
        p2, p98 = np.percentile(valid, [2, 98])
        normalized = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
        normalized[np.isnan(normalized)] = 0
        
        return normalized
    
    @staticmethod
    def _export_statistics(data_ctx, engine, final_mask, out_dir: str):
        """
        Export detection statistics to text file.
        
        Parameters:
        -----------
        data_ctx : GeoDataContext
            Data context
        engine : FusionEngine
            Fusion engine
        final_mask : numpy.ndarray
            Final mask
        out_dir : str
            Output directory
        """
        import os
        
        stats_file = os.path.join(out_dir, 'detection_statistics.txt')
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("遥感异常探测统计报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"矿产类型: {data_ctx.mineral_type}\n")
            f.write(f"Levashov模式: {'启用' if data_ctx.levashov_mode else '禁用'}\n")
            f.write(f"数据目录: {data_ctx.data_dir}\n\n")
            
            # ROI statistics
            total_pixels = np.sum(data_ctx.inROI)
            anomaly_pixels = np.sum(final_mask > 0)
            anomaly_ratio = anomaly_pixels / total_pixels * 100 if total_pixels > 0 else 0
            
            f.write("区域统计:\n")
            f.write(f"  总像素数: {total_pixels}\n")
            f.write(f"  异常像素数: {anomaly_pixels}\n")
            f.write(f"  异常比例: {anomaly_ratio:.2f}%\n\n")
            
            # Detector statistics
            f.write("各探测器结果:\n")
            for name, result in engine.results.items():
                mask = result['mask']
                det_anomaly = np.sum(mask > 0)
                det_ratio = det_anomaly / total_pixels * 100 if total_pixels > 0 else 0
                f.write(f"  {name}: {det_anomaly} 像素 ({det_ratio:.2f}%)\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"    ✓ 保存统计报告: {stats_file}")
