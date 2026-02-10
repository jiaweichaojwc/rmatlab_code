"""
Visualizer utility for creating plots and figures.
可视化工具 - 用于创建图表和图形
"""

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """
    Utility class for creating visualizations.
    用于创建可视化的工具类
    """
    
    @staticmethod
    def plot_anomaly_map(data, mask, title="Anomaly Detection", output_path=None):
        """
        Plot anomaly detection results.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Background data to display
        mask : numpy.ndarray
            Binary anomaly mask
        title : str
            Plot title
        output_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot background
        im = ax.imshow(data, cmap='gray', interpolation='nearest')
        
        # Overlay mask
        mask_overlay = np.ma.masked_where(mask == 0, mask)
        ax.imshow(mask_overlay, cmap='hot', alpha=0.6, interpolation='nearest')
        
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Intensity')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_multiple_detectors(results_dict, output_path=None):
        """
        Plot results from multiple detectors.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary of detector results
        output_path : str, optional
            Path to save figure
        """
        n_detectors = len(results_dict)
        fig, axes = plt.subplots(1, n_detectors, figsize=(5*n_detectors, 5))
        
        if n_detectors == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(results_dict.items()):
            mask = result['mask']
            axes[idx].imshow(mask, cmap='hot', interpolation='nearest')
            axes[idx].set_title(name)
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
