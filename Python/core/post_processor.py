"""
PostProcessor - Final stage processing for mineral detection pipeline.

This module performs post-processing on detector results including:
- Safe extraction of detector results with defaults
- Yakymchuk depth and pressure calculations
- Surface potential calculation using ASTER bands and PCA
- Gaussian filtering and fusion weight boosting
- Top 20 anomaly points extraction
- RGB image generation
- Comprehensive visualizations
- MAT file export and KMZ generation
"""

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

# Import internal modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.geo_utils import GeoUtils
from utils.visualizer import Visualizer
from utils.export_kmz import export_kmz


class PostProcessor:
    """
    Static class for post-processing mineral detection results.
    
    This is the final stage that integrates all detector outputs and produces
    final predictions, visualizations, and exports.
    """
    
    @staticmethod
    def run(ctx: Any, engine: Any, final_mask: np.ndarray, out_dir: str) -> None:
        """
        Execute the complete post-processing pipeline.
        
        This method performs:
        1. Safe extraction of detector results
        2. Depth and pressure calculation using Yakymchuk parameters
        3. Surface potential calculation using ASTER PCA analysis
        4. Gaussian filtering (sigma=8, then sigma=6)
        5. Fusion weight boosting (40% for anomaly points)
        6. Top 20 points extraction
        7. RGB image generation
        8. Visualization generation (resonance, mask fusion, deep prediction)
        9. MAT file export using scipy.io.savemat
        10. KMZ export
        
        Args:
            ctx: GeoDataContext object containing all input data
            engine: FusionEngine object with detector results
            final_mask: Final integrated anomaly mask (H x W)
            out_dir: Output directory path for saving results
            
        Returns:
            None
            
        Example:
            >>> PostProcessor.run(ctx, engine, final_mask, './output/')
        """
        print('=== 进入后处理阶段 (4掩码增强版) ===')
        
        # ================= Helper function for safe result extraction =================
        def safe_get(name: str) -> Dict[str, Any]:
            """
            Safely retrieve detector results with default empty values if not found.
            
            Args:
                name: Name of the detector result to retrieve
                
            Returns:
                Dictionary with 'mask' and 'debug' fields
            """
            if name in engine.results:
                res = engine.results[name]
            else:
                # Return default empty results
                res = {
                    'mask': np.zeros(ctx.in_roi.shape, dtype=float),
                    'debug': {
                        'F_map': np.zeros(ctx.in_roi.shape, dtype=float),
                        'delta_red_edge': np.zeros(ctx.in_roi.shape, dtype=float),
                        'moran_local': np.zeros(ctx.in_roi.shape, dtype=float),
                        'F_abs': np.zeros(ctx.in_roi.shape, dtype=float)
                    }
                }
            return res
        
        # ================= 1. Extract detector results =================
        res_red = safe_get('RedEdge')
        res_int = safe_get('Intrinsic')
        res_slow = safe_get('SlowVars')
        res_known = safe_get('KnownAnomaly')
        
        # Extract masks
        anomaly_mask_rededge = res_red['mask']
        anomaly_mask_fabs = res_int['mask']
        anomaly_mask_slow = res_slow['mask']
        anomaly_mask_known = res_known['mask']
        
        # Extract debug data
        F_map = res_red['debug']['F_map']
        delta_red = res_red['debug']['delta_red_edge']
        moran_local = res_int['debug']['moran_local']
        F_abs = res_int['debug']['F_abs']
        
        # ================= 2. Depth and Pressure Calculation =================
        # Get Yakymchuk parameters for the mineral type
        params = GeoUtils.get_yakymchuk_params(ctx.mineral_type)
        
        # Physical constants
        c = 3e8  # Speed of light (m/s)
        epsilon_r = 16  # Relative permittivity
        
        # Calculate resonance frequency using Yakymchuk model
        f_res_MHz = params['a'] + params['b'] * np.exp(-params['c'] * np.abs(F_map))
        
        # Handle NaN and negative values
        f_res_MHz[np.isnan(f_res_MHz)] = params['a']
        f_res_MHz[f_res_MHz < 0] = params['a']
        f_res_MHz[~ctx.in_roi] = np.nan
        
        # Calculate depth map (convert to km)
        depth_map = c / (2 * f_res_MHz * 1e6 * np.sqrt(epsilon_r)) / 1000
        
        # Clamp depth to [0, 4] km
        depth_map = np.clip(depth_map, 0, 4)
        depth_map[~ctx.in_roi] = np.nan
        
        # Calculate pressure gradient
        grad_P = 25 + 5 * depth_map
        grad_P = np.clip(grad_P, 0, 40)
        grad_P[~ctx.in_roi] = np.nan
        
        # ================= 3. Surface Potential Calculation =================
        # Get mineral-specific enhancement function
        _, _, _, enh_func = GeoUtils.get_mineral_thresholds(ctx.mineral_type)
        
        # Small epsilon to avoid division by zero
        eps_val = 1e-6
        
        # Calculate spectral indices
        Ferric = GeoUtils.mat2gray_roi(
            ctx.ast[:, :, 1] / (ctx.ast[:, :, 0] + eps_val),  # Band 2/Band 1 (0-indexed)
            ctx.in_roi
        )
        
        Clay = GeoUtils.mat2gray_roi(
            ctx.ast[:, :, 5] / (ctx.ast[:, :, 6] + eps_val),  # Band 6/Band 7 (0-indexed)
            ctx.in_roi
        )
        
        NDVI_inv = GeoUtils.mat2gray_roi(
            1 - (ctx.nir - ctx.red) / (ctx.nir + ctx.red + eps_val),
            ctx.in_roi
        )
        
        # ================= 4. PCA Analysis on ASTER bands 4-7 =================
        # Extract bands 4-7 (indices 3-6 in 0-indexed)
        pca_input = ctx.ast[:, :, 3:7]
        H, W, _ = pca_input.shape
        
        # Reshape to 2D for PCA
        pca_input_2d = pca_input.reshape(H * W, 4)
        
        # Standardize data
        pca_mean = np.nanmean(pca_input_2d, axis=0)
        pca_std = np.nanstd(pca_input_2d, axis=0)
        pca_input_2d = (pca_input_2d - pca_mean) / pca_std
        pca_input_2d[np.isnan(pca_input_2d)] = 0
        
        # Perform PCA using SVD (equivalent to sklearn's PCA)
        # Center the data
        data_centered = pca_input_2d - pca_input_2d.mean(axis=0)
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        
        # Compute scores (principal components)
        score = U @ np.diag(S)
        
        # Reshape back to 3D
        pca_result = score.reshape(H, W, 4)
        
        # Extract PCA components
        Hydroxy_anomaly = GeoUtils.mat2gray_roi(pca_result[:, :, 1], ctx.in_roi)  # PC2
        Fe_anomaly = GeoUtils.mat2gray_roi(pca_result[:, :, 2], ctx.in_roi)  # PC3
        
        # ================= 5. Calculate surface potential =================
        if ctx.mineral_type.lower() == 'cave':
            # For cave type, include DEM indices
            dem_indices = GeoUtils.compute_dem_indices(ctx.dem, 'cave', H, W, ctx.in_roi)
            Au_surface = enh_func(
                Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv,
                dem_indices['slope'], dem_indices['neg_curvature']
            )
            Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.in_roi)
        else:
            # Standard enhancement
            Au_surface = enh_func(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv)
        
        # ================= 6. First Gaussian Filter (sigma=8) =================
        valid_mask = ctx.in_roi & ~np.isnan(Au_surface)
        Au_temp = Au_surface.copy()
        Au_temp[~valid_mask] = 0
        
        # Apply Gaussian filter with replicate padding
        Au_filt = gaussian_filter(Au_temp, sigma=8, mode='nearest')
        
        # Update only valid pixels
        Au_surface[valid_mask] = Au_filt[valid_mask]
        Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.in_roi)
        
        # ================= 7. Fusion Weight Boosting (40%) =================
        # Resize final_mask if needed
        if final_mask.shape != Au_surface.shape:
            from scipy.ndimage import zoom
            zoom_factors = (Au_surface.shape[0] / final_mask.shape[0],
                          Au_surface.shape[1] / final_mask.shape[1])
            final_mask = zoom(final_mask, zoom_factors, order=0)  # Nearest neighbor
        
        # Boost potential by 40% for anomaly points
        Au_surface[ctx.in_roi] = Au_surface[ctx.in_roi] * (1 + final_mask[ctx.in_roi] * 0.4)
        
        # Handle NaN and inf values
        Au_surface[ctx.in_roi & (np.isnan(Au_surface) | np.isinf(Au_surface))] = 0
        
        # ================= 8. Second Gaussian Filter (sigma=6) =================
        valid_mask = ctx.in_roi & ~np.isnan(Au_surface)
        Au_temp = Au_surface.copy()
        Au_temp[~valid_mask] = 0
        
        # Apply Gaussian filter
        Au_filt = gaussian_filter(Au_temp, sigma=6, mode='nearest')
        
        # Update only valid pixels
        Au_surface[valid_mask] = Au_filt[valid_mask]
        Au_deep = GeoUtils.mat2gray_roi(Au_surface, ctx.in_roi)
        
        # ================= 9. Extract Top 20 Points =================
        temp = Au_deep.copy()
        temp[~ctx.in_roi] = 0
        
        # Sort and get top 20 indices
        flat_indices = np.argsort(temp.ravel())[::-1]  # Descending order
        top20_indices = flat_indices[:min(20, len(flat_indices))]
        
        # Convert to 2D coordinates
        topY, topX = np.unravel_index(top20_indices, (H, W))
        
        # Flip latitude grid for correct orientation
        lat_grid_corrected = np.flipud(ctx.lat_grid)
        
        # Extract coordinates for top points
        lon_top = ctx.lon_grid.ravel()[top20_indices]
        lat_top = lat_grid_corrected.ravel()[top20_indices]
        red_idx = np.arange(len(top20_indices))
        
        # ================= 10. Generate RGB Image =================
        img_rgb = np.stack([
            GeoUtils.mat2gray_roi(ctx.red, ctx.in_roi),
            GeoUtils.mat2gray_roi(ctx.green, ctx.in_roi),
            GeoUtils.mat2gray_roi(ctx.blue, ctx.in_roi)
        ], axis=2)
        img_rgb[np.isnan(img_rgb)] = 0
        
        # ================= 11. Generate Visualizations =================
        # Resonance parameters plot
        Visualizer.run_resonance(
            F_map, delta_red, moran_local, final_mask,
            depth_map * 1000,  # Convert to meters
            grad_P, f_res_MHz, img_rgb,
            out_dir, ctx.lon_grid, ctx.lat_grid
        )
        
        # Mask fusion plot (5 masks)
        masks_pack = [
            anomaly_mask_rededge,
            anomaly_mask_fabs,
            anomaly_mask_slow,
            anomaly_mask_known,
            final_mask
        ]
        titles_pack = [
            '1.红边异常',
            '2.本征吸收',
            '3.慢变量突变',
            '4.已知KML异常',
            '5.集成并集'
        ]
        Visualizer.run_mask_fusion(
            masks_pack, titles_pack,
            ctx.lon_grid, ctx.lat_grid,
            out_dir
        )
        
        # Deep prediction plot
        Visualizer.run_deep_prediction(
            Au_deep, ctx.lon_grid, ctx.lat_grid,
            ctx.lon_roi, ctx.lat_roi,
            lon_top, lat_top, red_idx,
            ctx.mineral_type, out_dir
        )
        
        # ================= 12. Save MAT file =================
        data_file = os.path.join(out_dir, f'{ctx.mineral_type}_Result.mat')
        
        # Replace NaN with 0 for MATLAB compatibility
        Au_deep_save = Au_deep.copy()
        Au_deep_save[np.isnan(Au_deep_save)] = 0
        
        F_abs_save = F_abs.copy()
        F_abs_save[np.isnan(F_abs_save)] = 0
        
        depth_map_save = depth_map.copy()
        depth_map_save[np.isnan(depth_map_save)] = 0
        
        f_res_MHz_save = f_res_MHz.copy()
        f_res_MHz_save[np.isnan(f_res_MHz_save)] = 0
        
        moran_local_save = moran_local.copy()
        moran_local_save[np.isnan(moran_local_save)] = 0
        
        # Prepare data dictionary for saving
        save_data = {
            'Au_deep': Au_deep_save,
            'F_abs': F_abs_save,
            'anomaly_mask_fabs': anomaly_mask_fabs,
            'anomaly_mask_rededge': anomaly_mask_rededge,
            'anomaly_mask_known': anomaly_mask_known,
            'depth_map': depth_map_save,
            'f_res_MHz': f_res_MHz_save,
            'final_anomaly_mask': final_mask,
            'inROI': ctx.in_roi.astype(float),
            'latGrid': ctx.lat_grid,
            'lonGrid': ctx.lon_grid,
            'latROI': ctx.lat_roi,
            'lonROI': ctx.lon_roi,
            'latTop': lat_top,
            'lonTop': lon_top,
            'mineral_type': ctx.mineral_type,
            'moran_local': moran_local_save,
            'redIdx': red_idx
        }
        
        # Save to MAT file
        scipy.io.savemat(data_file, save_data, do_compression=True)
        print(f'✅ MAT file saved: {data_file}')
        
        # ================= 13. Export KMZ =================
        try:
            status, message = export_kmz(data_file, out_dir)
            if status == 0:
                print(f'✅ KMZ export successful')
            else:
                print(f'⚠️ KMZ export failed: {message}')
        except Exception as e:
            print(f'⚠️ KMZ export error: {e}')
        
        print('=== 后处理完成 ===')
