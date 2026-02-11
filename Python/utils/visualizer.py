"""
Visualizer module for generating high-resolution plots from analysis results.

This module provides static methods for creating various visualization plots
including resonance parameters, mask fusion, and deep prediction contours.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Any
import os
from pathlib import Path


class Visualizer:
    """Static class for creating analysis visualization plots."""
    
    @staticmethod
    def run_resonance(
        F_map: np.ndarray,
        delta_red: np.ndarray,
        moran: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        gradP: np.ndarray,
        freq: np.ndarray,
        RGB: np.ndarray,
        outDir: str,
        lonGrid: np.ndarray,
        latGrid: np.ndarray
    ) -> None:
        """
        Generate a comprehensive 2x4 grid of resonance parameter plots.
        
        Args:
            F_map: F-statistic discrimination map
            delta_red: Red edge shift values
            moran: Moran's I spatial autocorrelation
            mask: Integrated anomaly mask
            depth: Depth values (in meters, will be converted to mm)
            gradP: Pressure gradient values
            freq: Frequency values
            RGB: RGB composite image
            outDir: Output directory path
            lonGrid: Longitude grid
            latGrid: Latitude grid
        """
        try:
            # Create longitude and latitude vectors
            lonV = np.linspace(lonGrid.min(), lonGrid.max(), F_map.shape[1])
            latV = np.linspace(latGrid.min(), latGrid.max(), F_map.shape[0])
            
            # Create figure with 2x4 grid
            fig = plt.figure(figsize=(20, 12), facecolor='w')
            fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, 
                              wspace=0.2, hspace=0.25)
            
            # Prepare data and metadata
            data_list = [RGB, F_map, delta_red, moran, mask, depth * 1000, gradP, freq]
            titles = ['RGB', 'F判别', '红边位移', 'Moran I', '综合异常', '深度', '压力', '频率']
            clims = [None, (0, 0.15), (-15, 15), (0, 1), (0, 1), (0, 2000), (0, 40), (10, 100)]
            
            # Create each subplot
            for i in range(8):
                ax = plt.subplot(2, 4, i + 1)
                data = np.flipud(data_list[i])
                
                im = ax.imshow(data, extent=[lonV.min(), lonV.max(), latV.min(), latV.max()],
                             aspect='auto', origin='lower', interpolation='nearest')
                
                ax.set_title(titles[i], fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                # Apply colormap and colorbar for non-RGB images
                if i > 0:
                    im.set_cmap('turbo')
                    if clims[i] is not None:
                        im.set_clim(clims[i])
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Save figure
            output_path = os.path.join(outDir, '01_共振参数综合图.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='w')
            plt.close(fig)
            
            print(f"✅ Resonance plot saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error in run_resonance: {e}")
            raise
    
    @staticmethod
    def run_mask_fusion(
        mask_list: List[np.ndarray],
        title_list: List[str],
        lonGrid: np.ndarray,
        latGrid: np.ndarray,
        outDir: str
    ) -> None:
        """
        Generate dynamic layout for 4-5 mask images with custom colormap.
        
        Args:
            mask_list: List of mask arrays to visualize
            title_list: List of titles for each mask
            lonGrid: Longitude grid
            latGrid: Latitude grid
            outDir: Output directory path
        """
        try:
            num_masks = len(mask_list)
            if num_masks == 0:
                print("⚠️ No masks to visualize")
                return
            
            # Create longitude and latitude vectors
            lonV = np.linspace(lonGrid.min(), lonGrid.max(), mask_list[0].shape[1])
            latV = np.linspace(latGrid.min(), latGrid.max(), mask_list[0].shape[0])
            
            # Smart layout: <=4 in one row, >4 use multiple rows
            if num_masks <= 4:
                cols = num_masks
                rows = 1
            else:
                cols = 4
                rows = int(np.ceil(num_masks / cols))
            
            # Create figure with dynamic size
            fig_width = 4 * cols
            fig_height = 4 * rows + 0.5
            fig = plt.figure(figsize=(fig_width, fig_height), facecolor='w')
            fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                              wspace=0.15, hspace=0.2)
            
            # Create custom colormap: light blue -> red
            n_colors = 256
            custom_cmap = np.column_stack([
                np.linspace(1, 0, n_colors),      # R: 1 -> 0
                np.linspace(1, 0.8, n_colors),    # G: 1 -> 0.8
                np.linspace(1, 0, n_colors)       # B: 1 -> 0
            ])
            custom_cmap = matplotlib.colors.ListedColormap(custom_cmap)
            
            # Plot each mask
            for i in range(num_masks):
                ax = plt.subplot(rows, cols, i + 1)
                
                # Prepare data
                img_data = mask_list[i].astype(float)
                img_data[np.isnan(img_data)] = 0
                img_data = np.flipud(img_data)
                
                # Display image
                im = ax.imshow(img_data, extent=[lonV.min(), lonV.max(), latV.min(), latV.max()],
                             aspect='auto', origin='lower', cmap=custom_cmap,
                             vmin=0, vmax=1, interpolation='nearest')
                
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                # Add title
                if i < len(title_list):
                    ax.set_title(title_list[i], fontsize=12, fontweight='bold')
            
            # Save figure
            output_path = os.path.join(outDir, f'02_掩码集成_{num_masks}图.png')
            plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='w')
            plt.close(fig)
            
            print(f"✅ Mask fusion plot saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error in run_mask_fusion: {e}")
            raise
    
    @staticmethod
    def run_deep_prediction(
        Au: np.ndarray,
        lonG: np.ndarray,
        latG: np.ndarray,
        lonR: np.ndarray,
        latR: np.ndarray,
        lonT: Optional[np.ndarray],
        latT: Optional[np.ndarray],
        rIdx: Optional[np.ndarray],
        mineral: str,
        outDir: str
    ) -> None:
        """
        Generate deep prediction contour plot with ROI and top points.
        
        Args:
            Au: Prediction values (probability or concentration)
            lonG: Longitude grid
            latG: Latitude grid
            lonR: ROI boundary longitudes
            latR: ROI boundary latitudes
            lonT: Top points longitudes (optional)
            latT: Top points latitudes (optional)
            rIdx: Indices of recommended top points (optional)
            mineral: Mineral name for title
            outDir: Output directory path
        """
        try:
            # Create longitude and latitude vectors
            lonV = np.linspace(lonG.min(), lonG.max(), Au.shape[1])
            latV = np.linspace(latG.min(), latG.max(), Au.shape[0])
            
            # Create figure
            fig = plt.figure(figsize=(12, 10), facecolor='w')
            ax = plt.gca()
            
            # Flip data for correct orientation
            Au_flipped = np.flipud(Au.astype(float))
            
            # Create filled contour plot
            levels = np.linspace(0.4, 1.0, 80)
            contourf = ax.contourf(lonV, latV, Au_flipped, levels=levels,
                                  cmap='jet', extend='both')
            
            # Add colorbar
            cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
            contourf.set_clim(0.4, 1.0)
            
            # Add contour lines
            contour_levels = np.arange(0.4, 1.05, 0.05)
            ax.contour(lonV, latV, Au_flipped, levels=contour_levels,
                      colors='lightgray', linewidths=0.5, alpha=0.7)
            
            # Plot ROI boundary
            ax.plot(lonR, latR, 'k-', linewidth=2.5, label='ROI Boundary')
            
            # Plot top points if provided
            if lonT is not None and latT is not None and len(lonT) > 0:
                ax.plot(lonT, latT, 'wo', markersize=10, markerfacecolor='#333333',
                       markeredgewidth=1.5, markeredgecolor='white', label='Top Points')
            
            # Highlight recommended points if provided
            if rIdx is not None and len(rIdx) > 0 and lonT is not None and latT is not None:
                ax.plot(lonT[rIdx], latT[rIdx], 'yo', markersize=18, 
                       markerfacecolor='none', markeredgewidth=3, 
                       markeredgecolor='yellow', label='Recommended')
            
            # Configure plot
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Deep Prediction: {mineral.upper()}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            if (lonT is not None and len(lonT) > 0) or (rIdx is not None and len(rIdx) > 0):
                ax.legend(loc='best', framealpha=0.9)
            
            # Save as PNG
            png_path = os.path.join(outDir, '03_深部成矿预测图.png')
            plt.savefig(png_path, dpi=500, bbox_inches='tight', facecolor='w')
            
            print(f"✅ Deep prediction plot saved: {png_path}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"❌ Error in run_deep_prediction: {e}")
            raise
