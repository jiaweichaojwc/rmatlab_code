"""
Visualizer - Visualization tools for mineral detection results.

This module provides static methods for generating publication-quality
visualizations of resonance parameters, mask fusion, and deep prediction results.
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings

warnings.filterwarnings('ignore')


class Visualizer:
    """Static class for visualization of mineral detection results."""
    
    @staticmethod
    def run_resonance(
        f_map: np.ndarray,
        delta_red: np.ndarray,
        moran: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        grad_p: np.ndarray,
        freq: np.ndarray,
        rgb: np.ndarray,
        out_dir: str,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray
    ) -> None:
        """
        Generate comprehensive resonance parameter visualization.
        
        Creates a 2x4 grid showing:
        - RGB composite
        - F-statistic map
        - Red edge shift
        - Moran's I
        - Integrated anomaly mask
        - Depth estimate
        - Pressure gradient
        - Resonance frequency
        
        Args:
            f_map: F-statistic discrimination map
            delta_red: Red edge position shift
            moran: Moran's I local spatial autocorrelation
            mask: Integrated anomaly mask
            depth: Depth map (in meters)
            grad_p: Pressure gradient map
            freq: Resonance frequency map (in MHz)
            rgb: RGB composite image
            out_dir: Output directory
            lon_grid: Longitude grid
            lat_grid: Latitude grid
        """
        # Create coordinate vectors
        lon_v = np.linspace(lon_grid.min(), lon_grid.max(), f_map.shape[1])
        lat_v = np.linspace(lat_grid.min(), lat_grid.max(), f_map.shape[0])
        
        # Create figure
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.25)
        
        # Data and visualization parameters
        data_list = [rgb, f_map, delta_red, moran, mask, depth, grad_p, freq]
        titles = ['RGB', 'F判别', '红边位移', 'Moran I', '综合异常', '深度', '压力', '频率']
        clims = [None, (0, 0.15), (-15, 15), (0, 1), (0, 1), (0, 2000), (0, 40), (10, 100)]
        
        for i in range(8):
            ax = plt.subplot(2, 4, i + 1)
            
            data = data_list[i]
            if i == 0:  # RGB image
                ax.imshow(np.flipud(data), extent=[lon_v[0], lon_v[-1], lat_v[0], lat_v[-1]],
                         aspect='auto', origin='lower')
            else:
                im = ax.imshow(np.flipud(data), extent=[lon_v[0], lon_v[-1], lat_v[0], lat_v[-1]],
                              aspect='auto', origin='lower', cmap='turbo')
                
                # Set color limits if specified
                if clims[i] is not None:
                    im.set_clim(clims[i])
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Save figure
        output_path = os.path.join(out_dir, '01_共振参数综合图.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved resonance visualization to: {output_path}')
    
    @staticmethod
    def run_mask_fusion(
        mask_list: List[np.ndarray],
        title_list: List[str],
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        out_dir: str
    ) -> None:
        """
        Generate mask fusion visualization showing all detector masks.
        
        Creates an adaptive grid layout showing 4-5 masks in optimal arrangement.
        
        Args:
            mask_list: List of mask arrays (binary or float [0-1])
            title_list: List of titles for each mask
            lon_grid: Longitude grid
            lat_grid: Latitude grid
            out_dir: Output directory
        """
        num_masks = len(mask_list)
        if num_masks == 0:
            return
        
        # Create coordinate vectors
        lon_v = np.linspace(lon_grid.min(), lon_grid.max(), mask_list[0].shape[1])
        lat_v = np.linspace(lat_grid.min(), lat_grid.max(), mask_list[0].shape[0])
        
        # Smart layout: <=4 in one row, >4 wrap to 2 rows
        if num_masks <= 4:
            cols = num_masks
            rows = 1
        else:
            cols = 4
            rows = int(np.ceil(num_masks / cols))
        
        # Create figure
        fig = plt.figure(figsize=(4 * cols, 4 * rows + 0.5), facecolor='white')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15, hspace=0.25)
        
        # Custom colormap: light blue -> red
        n_colors = 256
        custom_cmap = colors.LinearSegmentedColormap.from_list(
            'custom',
            [(1, 1, 1), (1, 0.8, 0)]  # White to orange-red
        )
        
        for i in range(num_masks):
            ax = plt.subplot(rows, cols, i + 1)
            
            # Prepare data
            img_data = mask_list[i].astype(np.float64)
            img_data = np.nan_to_num(img_data, 0)
            
            # Display mask
            im = ax.imshow(np.flipud(img_data), extent=[lon_v[0], lon_v[-1], lat_v[0], lat_v[-1]],
                          aspect='auto', origin='lower', cmap=custom_cmap, vmin=0, vmax=1)
            
            # Title
            if i < len(title_list):
                ax.set_title(title_list[i], fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Save figure
        output_name = os.path.join(out_dir, f'02_掩码集成_{num_masks}图.png')
        plt.savefig(output_name, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved mask fusion visualization to: {output_name}')
    
    @staticmethod
    def run_deep_prediction(
        au: np.ndarray,
        lon_g: np.ndarray,
        lat_g: np.ndarray,
        lon_r: np.ndarray,
        lat_r: np.ndarray,
        lon_t: np.ndarray,
        lat_t: np.ndarray,
        r_idx: np.ndarray,
        mineral: str,
        out_dir: str
    ) -> None:
        """
        Generate deep mineralization prediction visualization.
        
        Creates a contour map with top anomalies highlighted and ROI boundary.
        
        Args:
            au: Deep prediction map (0-1 normalized)
            lon_g: Longitude grid
            lat_g: Latitude grid
            lon_r: ROI boundary longitude coordinates
            lat_r: ROI boundary latitude coordinates
            lon_t: Top anomaly longitude coordinates
            lat_t: Top anomaly latitude coordinates
            r_idx: Indices of red-highlighted top anomalies
            mineral: Mineral type name
            out_dir: Output directory
        """
        # Create coordinate vectors
        lon_v = np.linspace(lon_g.min(), lon_g.max(), au.shape[1])
        lat_v = np.linspace(lat_g.min(), lat_g.max(), au.shape[0])
        
        # Create figure
        fig = plt.figure(figsize=(12, 10), facecolor='white')
        ax = plt.subplot(111)
        
        # Create meshgrid for contour plotting
        lon_mesh, lat_mesh = np.meshgrid(lon_v, lat_v)
        
        # Filled contour plot
        contour_levels = 80
        cf = ax.contourf(lon_mesh, np.flipud(lat_mesh), np.flipud(au.astype(np.float64)),
                        levels=contour_levels, cmap='jet', vmin=0.4, vmax=1.0)
        
        # Add colorbar
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mineralization Potential', fontsize=12)
        
        # Contour lines
        contour_lines = np.arange(0.4, 1.05, 0.05)
        cs = ax.contour(lon_mesh, np.flipud(lat_mesh), np.flipud(au.astype(np.float64)),
                       levels=contour_lines, colors='white', linewidths=0.5, alpha=0.7)
        
        # ROI boundary
        if lon_r is not None and lat_r is not None and len(lon_r) > 0:
            ax.plot(lon_r, lat_r, 'k-', linewidth=2.5, label='ROI Boundary')
        
        # Top anomalies (all)
        if lon_t is not None and lat_t is not None and len(lon_t) > 0:
            ax.plot(lon_t, lat_t, 'wo', markersize=10, markerfacecolor=[0.2, 0.2, 0.2],
                   markeredgecolor='white', markeredgewidth=1.5, label='Top Anomalies')
        
        # Highlighted top anomalies (red)
        if r_idx is not None and len(r_idx) > 0 and lon_t is not None:
            ax.plot(lon_t[r_idx], lat_t[r_idx], 'yo', markersize=18, 
                   markerfacecolor='none', markeredgecolor='yellow', markeredgewidth=3,
                   label='Top Priority')
        
        # Labels and formatting
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Deep Prediction: {mineral.upper()}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if lon_t is not None or lon_r is not None:
            ax.legend(loc='upper right', fontsize=10)
        
        # Save both PNG and as pickle for later editing
        output_png = os.path.join(out_dir, '03_深部成矿预测图.png')
        output_pkl = os.path.join(out_dir, '03_深部成矿预测图.pkl')
        
        plt.savefig(output_png, dpi=500, bbox_inches='tight')
        
        # Save figure object for potential later editing
        try:
            import pickle
            with open(output_pkl, 'wb') as f:
                pickle.dump(fig, f)
        except Exception as e:
            print(f'Warning: Could not save figure as pickle: {e}')
        
        plt.close(fig)
        print(f'Saved deep prediction visualization to: {output_png}')
