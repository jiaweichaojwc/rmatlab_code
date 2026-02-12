"""
PostProcessor - Post-processing stage for mineral detection results.

This module handles the final processing stage including:
- Multi-mask integration and enhancement
- Depth and pressure estimation from resonance data
- Surface potential calculation
- Visualization generation
- Results export to MAT files
"""

import sys
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy import ndimage
from scipy.io import savemat
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Utils.geo_utils import GeoUtils
from Utils.visualizer import Visualizer


class PostProcessor:
    """Static class for post-processing mineral detection results."""
    
    @staticmethod
    def safe_get(engine, name: str, in_roi: np.ndarray) -> Dict[str, Any]:
        """
        Safely retrieve detector result with default values if not present.
        
        Args:
            engine: Detection engine containing results
            name: Name of the detector result
            in_roi: ROI mask for creating default-sized arrays
            
        Returns:
            Dictionary with mask and debug data
        """
        if name in engine.results:
            return engine.get_result(name)
        else:
            # Return default empty result
            return {
                'mask': np.zeros(in_roi.shape, dtype=np.float32),
                'debug': {
                    'F_map': np.zeros(in_roi.shape, dtype=np.float32),
                    'delta_red_edge': np.zeros(in_roi.shape, dtype=np.float32),
                    'moran_local': np.zeros(in_roi.shape, dtype=np.float32),
                    'F_abs': np.zeros(in_roi.shape, dtype=np.float32)
                }
            }
    
    @staticmethod
    def run(ctx, engine, final_mask: np.ndarray, out_dir: str) -> None:
        """
        Main post-processing workflow.
        
        Args:
            ctx: GeoDataContext containing all input data
            engine: FusionEngine containing detector results
            final_mask: Final integrated anomaly mask
            out_dir: Output directory for saving results
        """
        print('=== 进入后处理阶段 (多掩码增强版) ===')
        
        # Retrieve results from different detectors
        res_red = PostProcessor.safe_get(engine, 'RedEdge', ctx.in_roi)
        res_int = PostProcessor.safe_get(engine, 'Intrinsic', ctx.in_roi)
        res_slow = PostProcessor.safe_get(engine, 'SlowVars', ctx.in_roi)
        res_known = PostProcessor.safe_get(engine, 'KnownAnomaly', ctx.in_roi)
        
        anomaly_mask_rededge = res_red['mask']
        anomaly_mask_fabs = res_int['mask']
        anomaly_mask_slow = res_slow['mask']
        anomaly_mask_known = res_known['mask']
        
        # Debug data
        F_map = res_red['debug']['F_map']
        delta_red = res_red['debug']['delta_red_edge']
        moran_local = res_int['debug']['moran_local']
        F_abs = res_int['debug']['F_abs']
        
        # 1. Depth and pressure estimation
        params = GeoUtils.get_yakymchuk_params(ctx.mineral_type)
        c = 3e8  # Speed of light
        epsilon_r = 16  # Relative permittivity
        
        # Resonance frequency calculation
        f_res_mhz = params['a'] + params['b'] * np.exp(-params['c'] * np.abs(F_map))
        f_res_mhz = np.nan_to_num(f_res_mhz, nan=params['a'])
        f_res_mhz = np.clip(f_res_mhz, params['a'], None)
        f_res_mhz[~ctx.in_roi] = np.nan
        
        # Depth map calculation (in km)
        depth_map = c / (2 * f_res_mhz * 1e6 * np.sqrt(epsilon_r)) / 1000
        depth_map = np.clip(depth_map, 0, 4)
        depth_map[~ctx.in_roi] = np.nan
        
        # Pressure gradient estimation
        grad_p = 25 + 5 * depth_map
        grad_p = np.clip(grad_p, 0, 40)
        grad_p[~ctx.in_roi] = np.nan
        
        # 2. Surface potential calculation
        _, _, _, enh_func = GeoUtils.get_mineral_thresholds(ctx.mineral_type)
        eps_val = 1e-6
        h, w = ctx.ast.shape[:2]
        
        # Calculate spectral ratios
        ferric = GeoUtils.mat2gray_roi(
            ctx.ast[:, :, 2] / (ctx.ast[:, :, 1] + eps_val), ctx.in_roi
        )
        clay = GeoUtils.mat2gray_roi(
            ctx.ast[:, :, 6] / (ctx.ast[:, :, 7] + eps_val), ctx.in_roi
        )
        ndvi_inv = GeoUtils.mat2gray_roi(
            1 - (ctx.nir - ctx.red) / (ctx.nir + ctx.red + eps_val), ctx.in_roi
        )
        
        # PCA on ASTER bands 4-7
        pca_input = ctx.ast[:, :, 4:8].reshape(h * w, 4).astype(np.float64)
        pca_input = (pca_input - np.nanmean(pca_input, axis=0)) / np.nanstd(pca_input, axis=0)
        pca_input = np.nan_to_num(pca_input, 0)
        
        # Perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=4)
        score = pca.fit_transform(pca_input)
        pca_result = score.reshape(h, w, 4)
        
        hydroxy_anomaly = GeoUtils.mat2gray_roi(pca_result[:, :, 1], ctx.in_roi)
        fe_anomaly = GeoUtils.mat2gray_roi(pca_result[:, :, 2], ctx.in_roi)
        
        # Mineral-specific surface potential calculation
        if ctx.mineral_type.lower() == 'cave':
            dem_indices = GeoUtils.compute_dem_indices(ctx.dem, 'cave', h, w, ctx.in_roi)
            au_surface = enh_func(
                ferric, fe_anomaly, hydroxy_anomaly, clay, ndvi_inv,
                dem_indices['slope'], dem_indices['neg_curvature']
            )
            au_surface = GeoUtils.mat2gray_roi(au_surface, ctx.in_roi)
            
        elif ctx.mineral_type.lower() == 'offshore_petroleum':
            osi = GeoUtils.mat2gray_roi(
                (ctx.blue + ctx.green + ctx.red) / (ctx.nir + eps_val), ctx.in_roi
            )
            
            # Check for SAR dark spot data
            if hasattr(ctx, 'sar_dark_spot') and ctx.sar_dark_spot is not None:
                sds = ctx.sar_dark_spot
            else:
                sds = np.zeros((h, w), dtype=np.float32)
            
            au_surface = enh_func(
                ferric, fe_anomaly, hydroxy_anomaly, clay, ndvi_inv, osi, sds
            )
            au_surface = GeoUtils.mat2gray_roi(au_surface, ctx.in_roi)
        else:
            au_surface = enh_func(ferric, fe_anomaly, hydroxy_anomaly, clay, ndvi_inv)
        
        # Filter 1: First Gaussian smoothing
        valid_mask = ctx.in_roi & ~np.isnan(au_surface)
        au_temp = au_surface.copy()
        au_temp[~valid_mask] = 0
        au_filt = ndimage.gaussian_filter(au_temp, sigma=8, mode='nearest')
        au_surface[valid_mask] = au_filt[valid_mask]
        au_surface = GeoUtils.mat2gray_roi(au_surface, ctx.in_roi)
        
        # Fusion with final mask
        if final_mask.shape != au_surface.shape:
            from skimage.transform import resize
            final_mask = resize(final_mask, au_surface.shape, order=0, preserve_range=True)
        
        au_surface[ctx.in_roi] = au_surface[ctx.in_roi] * (1 + final_mask[ctx.in_roi] * 0.4)
        au_surface[ctx.in_roi & (np.isnan(au_surface) | np.isinf(au_surface))] = 0
        
        # Filter 2: Second Gaussian smoothing
        valid_mask = ctx.in_roi & ~np.isnan(au_surface)
        au_temp = au_surface.copy()
        au_temp[~valid_mask] = 0
        au_filt = ndimage.gaussian_filter(au_temp, sigma=6, mode='nearest')
        au_surface[valid_mask] = au_filt[valid_mask]
        au_deep = GeoUtils.mat2gray_roi(au_surface, ctx.in_roi)
        
        # 3. Top 20 anomalies
        temp = au_deep.copy()
        temp[~ctx.in_roi] = 0
        idx = np.argsort(temp.ravel())[::-1]
        top20 = idx[:min(20, len(idx))]
        top_y, top_x = np.unravel_index(top20, (h, w))
        
        lat_grid_corrected = np.flipud(ctx.lat_grid)
        lon_top = ctx.lon_grid.ravel()[top20]
        lat_top = lat_grid_corrected.ravel()[top20]
        red_idx = np.arange(len(top20))
        
        # 4. Visualization
        img_rgb = np.stack([
            GeoUtils.mat2gray_roi(ctx.red, ctx.in_roi),
            GeoUtils.mat2gray_roi(ctx.green, ctx.in_roi),
            GeoUtils.mat2gray_roi(ctx.blue, ctx.in_roi)
        ], axis=2)
        img_rgb = np.nan_to_num(img_rgb, 0)
        
        Visualizer.run_resonance(
            F_map, delta_red, moran_local, final_mask,
            depth_map * 1000, grad_p, f_res_mhz, img_rgb,
            out_dir, ctx.lon_grid, ctx.lat_grid
        )
        
        masks_pack = [
            anomaly_mask_rededge, anomaly_mask_fabs, anomaly_mask_slow,
            anomaly_mask_known, final_mask
        ]
        titles_pack = [
            '1.红边异常', '2.本征吸收', '3.慢变量突变', '4.已知KML异常', '5.集成并集'
        ]
        Visualizer.run_mask_fusion(
            masks_pack, titles_pack, ctx.lon_grid, ctx.lat_grid, out_dir
        )
        
        Visualizer.run_deep_prediction(
            au_deep, ctx.lon_grid, ctx.lat_grid,
            ctx.lon_roi, ctx.lat_roi, lon_top, lat_top, red_idx,
            ctx.mineral_type, out_dir
        )
        
        # 5. Save results to MAT file
        data_file = os.path.join(out_dir, f'{ctx.mineral_type}_Result.mat')
        
        # Replace NaN with 0 for saving
        au_deep = np.nan_to_num(au_deep, 0)
        F_abs = np.nan_to_num(F_abs, 0)
        depth_map = np.nan_to_num(depth_map, 0)
        f_res_mhz = np.nan_to_num(f_res_mhz, 0)
        moran_local = np.nan_to_num(moran_local, 0)
        
        # Prepare data dictionary
        save_dict = {
            'Au_deep': au_deep,
            'F_abs': F_abs,
            'anomaly_mask_fabs': anomaly_mask_fabs,
            'anomaly_mask_rededge': anomaly_mask_rededge,
            'anomaly_mask_slow': anomaly_mask_slow,
            'anomaly_mask_known': anomaly_mask_known,
            'depth_map': depth_map,
            'f_res_MHz': f_res_mhz,
            'final_anomaly_mask': final_mask,
            'inROI': ctx.in_roi,
            'latGrid': ctx.lat_grid,
            'lonGrid': ctx.lon_grid,
            'latROI': ctx.lat_roi,
            'lonROI': ctx.lon_roi,
            'latTop': lat_top,
            'lonTop': lon_top,
            'mineral_type': ctx.mineral_type,
            'moran_local': moran_local,
            'redIdx': red_idx
        }
        
        savemat(data_file, save_dict)
        print(f'Results saved to: {data_file}')
        
        # Export KMZ (placeholder for now)
        # export_kmz(data_file, out_dir)
