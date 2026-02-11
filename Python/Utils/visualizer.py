"""
Visualizer - Visualization utility for results
Converted from MATLAB Visualizer.m
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os


class Visualizer:
    """Static visualization utilities"""
    
    @staticmethod
    def run_resonance(F_map, delta_red, moran, mask, depth, gradP, freq, RGB, 
                     outDir, lonGrid, latGrid):
        """Generate resonance parameter comprehensive figure"""
        lonV = np.linspace(np.min(lonGrid), np.max(lonGrid), F_map.shape[1])
        latV = np.linspace(np.min(latGrid), np.max(latGrid), F_map.shape[0])
        
        fig = plt.figure(figsize=(20, 12), facecolor='w')
        fig.patch.set_facecolor('white')
        
        data_list = [RGB, F_map, delta_red, moran, mask, depth*1000, gradP, freq]
        titles = ['RGB', 'F判别', '红边位移', 'Moran I', '综合异常', '深度', '压力', '频率']
        clims = [None, [0, 0.15], [-15, 15], [0, 1], [0, 1], [0, 2000], [0, 40], [10, 100]]
        
        for i in range(8):
            ax = fig.add_subplot(2, 4, i+1)
            
            if i == 0:  # RGB
                ax.imshow(np.flipud(data_list[i]), extent=[lonV[0], lonV[-1], latV[0], latV[-1]], 
                         aspect='auto')
            else:
                im = ax.imshow(np.flipud(data_list[i]), extent=[lonV[0], lonV[-1], latV[0], latV[-1]], 
                             cmap='turbo', aspect='auto')
                cbar = plt.colorbar(im, ax=ax)
                if clims[i] is not None:
                    im.set_clim(clims[i])
            
            ax.set_title(titles[i])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, '01_共振参数综合图.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def run_mask_fusion(mask_list, title_list, lonGrid, latGrid, outDir):
        """Generate mask fusion figure with dynamic layout"""
        num_masks = len(mask_list)
        if num_masks == 0:
            return
        
        lonV = np.linspace(np.min(lonGrid), np.max(lonGrid), mask_list[0].shape[1])
        latV = np.linspace(np.min(latGrid), np.max(latGrid), mask_list[0].shape[0])
        
        # Smart layout: <=4 in one row, >4 wrap to next row
        if num_masks <= 4:
            cols = num_masks
            rows = 1
        else:
            cols = 4
            rows = int(np.ceil(num_masks / cols))
        
        fig = plt.figure(figsize=(4*cols, 4*rows + 0.5), facecolor='w')
        
        # Custom colormap: light blue -> red
        n_colors = 256
        custom_map = np.column_stack([
            np.linspace(1, 0, n_colors),
            np.linspace(1, 0.8, n_colors),
            np.linspace(1, 0, n_colors)
        ])
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(custom_map)
        
        for i in range(num_masks):
            ax = fig.add_subplot(rows, cols, i+1)
            
            img_data = mask_list[i].astype(float)
            img_data[np.isnan(img_data)] = 0
            
            im = ax.imshow(np.flipud(img_data), extent=[lonV[0], lonV[-1], latV[0], latV[-1]], 
                          cmap=cmap, vmin=0, vmax=1, aspect='auto')
            
            if i < len(title_list):
                ax.set_title(title_list[i], fontsize=12, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_name = os.path.join(outDir, f'02_掩码集成_{num_masks}图.png')
        plt.savefig(out_name, dpi=400, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def run_deep_prediction(Au, lonG, latG, lonR, latR, lonT, latT, rIdx, 
                           mineral, outDir):
        """Generate deep prediction figure"""
        # Create figure with visible=True to ensure rendering
        fig = plt.figure(figsize=(12, 10), facecolor='w')
        ax = fig.add_subplot(111)
        
        lonV = np.linspace(np.min(lonG), np.max(lonG), Au.shape[1])
        latV = np.linspace(np.min(latG), np.max(latG), Au.shape[0])
        
        # Contour fill
        cs = ax.contourf(lonV, latV, np.flipud(Au), levels=80, cmap='jet', 
                        vmin=0.4, vmax=1)
        plt.colorbar(cs, ax=ax)
        
        # Contour lines
        ax.contour(lonV, latV, np.flipud(Au), levels=np.arange(0.4, 1.05, 0.05), 
                  colors='lightgray', linewidths=0.5)
        
        # ROI boundary
        ax.plot(lonR, latR, 'k-', linewidth=2.5)
        
        # Target points
        if lonT is not None and len(lonT) > 0:
            ax.plot(lonT, latT, 'wo', markersize=10, markerfacecolor=[0.2, 0.2, 0.2])
        
        # Recommended points
        if rIdx is not None and len(rIdx) > 0:
            ax.plot(lonT[rIdx], latT[rIdx], 'yo', markersize=18, linewidth=3, 
                   markerfacecolor='none')
        
        ax.set_aspect('auto')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Deep Prediction: {mineral.upper()}', fontsize=16)
        
        # Force rendering
        plt.draw()
        
        # Save as .fig equivalent (pickle) and .png
        fig.savefig(os.path.join(outDir, '03_深部成矿预测图.png'), dpi=500, bbox_inches='tight')
        
        # Also save as pickle for later editing
        import pickle
        with open(os.path.join(outDir, '03_深部成矿预测图.pkl'), 'wb') as f:
            pickle.dump(fig, f)
        
        plt.close(fig)
