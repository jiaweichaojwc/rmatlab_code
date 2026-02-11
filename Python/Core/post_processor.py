"""
PostProcessor - Post-processing of detection results
Converted from MATLAB PostProcessor.m
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.geo_utils import GeoUtils
from Utils.visualizer import Visualizer
from Utils.export_kmz import exportKMZ


class PostProcessor:
    """Static post-processing utilities"""
    
    @staticmethod
    def run(ctx, engine, final_mask, outDir):
        """
        Run post-processing pipeline
        
        Parameters:
        -----------
        ctx : GeoDataContext
            Data context
        engine : FusionEngine
            Fusion engine with results
        final_mask : ndarray
            Final fused mask
        outDir : str
            Output directory
        """
        print('=== 进入后处理阶段 (4掩码增强版) ===')
        
        def safeGet(name):
            """Safely get detector result, return zeros if not found"""
            if name in engine.results:
                return engine.getResult(name)
            else:
                return {
                    'mask': np.zeros(ctx.inROI.shape),
                    'debug': {
                        'F_map': np.zeros(ctx.inROI.shape),
                        'delta_red_edge': np.zeros(ctx.inROI.shape),
                        'moran_local': np.zeros(ctx.inROI.shape),
                        'F_abs': np.zeros(ctx.inROI.shape)
                    }
                }
        
        res_Red = safeGet('RedEdge')
        res_Int = safeGet('Intrinsic')
        res_Slow = safeGet('SlowVars')
        res_Known = safeGet('KnownAnomaly')
        
        anomaly_mask_rededge = res_Red['mask']
        anomaly_mask_fabs = res_Int['mask']
        anomaly_mask_slow = res_Slow['mask']
        anomaly_mask_known = res_Known['mask']
        
        # Debug data
        F_map = res_Red['debug']['F_map']
        delta_red = res_Red['debug']['delta_red_edge']
        moran_local = res_Int['debug']['moran_local']
        F_abs = res_Int['debug']['F_abs']
        
        # 1. Depth and pressure calculation
        params = GeoUtils.getYakymchukParams(ctx.mineral_type)
        c = 3e8  # Speed of light
        epsilon_r = 16  # Relative permittivity
        
        f_res_MHz = params['a'] + params['b'] * np.exp(-params['c'] * np.abs(F_map))
        f_res_MHz[np.isnan(f_res_MHz)] = params['a']
        f_res_MHz[f_res_MHz < 0] = params['a']
        f_res_MHz[~ctx.inROI] = np.nan
        
        depth_map = c / (2 * f_res_MHz * 1e6 * np.sqrt(epsilon_r)) / 1000
        depth_map = np.clip(depth_map, 0, 4)
        depth_map[~ctx.inROI] = np.nan
        
        grad_P = 25 + 5 * depth_map
        grad_P = np.clip(grad_P, 0, 40)
        grad_P[~ctx.inROI] = np.nan
        
        # 2. Surface potential calculation
        _, _, _, enh_func = GeoUtils.getMineralThresholds(ctx.mineral_type)
        eps_val = 1e-6
        
        Ferric = GeoUtils.mat2gray_roi(
            ctx.ast[:, :, 1] / (ctx.ast[:, :, 0] + eps_val), ctx.inROI)
        Clay = GeoUtils.mat2gray_roi(
            ctx.ast[:, :, 5] / (ctx.ast[:, :, 6] + eps_val), ctx.inROI)
        NDVI_inv = GeoUtils.mat2gray_roi(
            1 - (ctx.NIR - ctx.Red) / (ctx.NIR + ctx.Red + eps_val), ctx.inROI)
        
        # PCA analysis
        H, W = ctx.ast.shape[:2]
        pcaInput = ctx.ast[:, :, 3:7].reshape(H * W, 4)  # B4-B7 (0-indexed: 3-6)
        pcaInput = (pcaInput - np.nanmean(pcaInput, axis=0)) / (np.nanstd(pcaInput, axis=0) + eps_val)
        pcaInput[np.isnan(pcaInput)] = 0
        
        pca = PCA(n_components=4)
        score = pca.fit_transform(pcaInput)
        pcaResult = score.reshape(H, W, 4)
        
        Hydroxy_anomaly = GeoUtils.mat2gray_roi(pcaResult[:, :, 1], ctx.inROI)
        Fe_anomaly = GeoUtils.mat2gray_roi(pcaResult[:, :, 2], ctx.inROI)
        
        # Calculate surface potential
        if ctx.mineral_type.lower() == 'cave':
            demIndices = GeoUtils.computeDEMIndices(ctx.dem, 'cave', H, W, ctx.inROI)
            Au_surface = enh_func(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv,
                                 demIndices['slope'], demIndices['neg_curvature'])
            Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI)
        else:
            Au_surface = enh_func(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv)
        
        # Filter 1
        valid_mask = ctx.inROI & ~np.isnan(Au_surface)
        Au_temp = Au_surface.copy()
        Au_temp[~valid_mask] = 0
        Au_filt = gaussian_filter(Au_temp, sigma=8, mode='constant', cval=0)
        Au_surface[valid_mask] = Au_filt[valid_mask]
        Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI)
        
        # Fusion weights (replicate untitled3.m logic)
        if final_mask.shape != Au_surface.shape:
            from scipy.ndimage import zoom
            zoom_factors = (Au_surface.shape[0] / final_mask.shape[0],
                           Au_surface.shape[1] / final_mask.shape[1])
            final_mask = zoom(final_mask, zoom_factors, order=0)
        
        # If point is in final anomaly mask, increase potential by 40%
        Au_surface[ctx.inROI] = Au_surface[ctx.inROI] * (1 + final_mask[ctx.inROI] * 0.4)
        Au_surface[ctx.inROI & (np.isnan(Au_surface) | np.isinf(Au_surface))] = 0
        
        # Filter 2
        valid_mask = ctx.inROI & ~np.isnan(Au_surface)
        Au_temp = Au_surface.copy()
        Au_temp[~valid_mask] = 0
        Au_filt = gaussian_filter(Au_temp, sigma=6, mode='constant', cval=0)
        Au_surface[valid_mask] = Au_filt[valid_mask]
        Au_deep = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI)
        
        # 3. Top 20 target locations
        temp = Au_deep.copy()
        temp[~ctx.inROI] = 0
        top_indices = np.argsort(temp.flatten())[::-1][:20]
        topY, topX = np.unravel_index(top_indices, (H, W))
        
        latGrid_corrected = np.flipud(ctx.latGrid)
        lonTop = ctx.lonGrid.flatten()[top_indices]
        latTop = latGrid_corrected.flatten()[top_indices]
        redIdx = np.arange(len(top_indices))
        
        # 4. Visualization
        img_rgb = np.stack([
            GeoUtils.mat2gray_roi(ctx.Red, ctx.inROI),
            GeoUtils.mat2gray_roi(ctx.Green, ctx.inROI),
            GeoUtils.mat2gray_roi(ctx.Blue, ctx.inROI)
        ], axis=2)
        img_rgb[np.isnan(img_rgb)] = 0
        
        Visualizer.run_resonance(
            F_map, delta_red, moran_local, final_mask,
            depth_map * 1000, grad_P, f_res_MHz, img_rgb,
            outDir, ctx.lonGrid, ctx.latGrid
        )
        
        # Pass 5 mask images
        masks_pack = [anomaly_mask_rededge, anomaly_mask_fabs, anomaly_mask_slow, 
                     anomaly_mask_known, final_mask]
        titles_pack = ['1.红边异常', '2.本征吸收', '3.慢变量突变', '4.已知KML异常', '5.集成并集']
        Visualizer.run_mask_fusion(masks_pack, titles_pack, ctx.lonGrid, ctx.latGrid, outDir)
        
        Visualizer.run_deep_prediction(
            Au_deep, ctx.lonGrid, ctx.latGrid,
            ctx.lonROI, ctx.latROI, lonTop, latTop, redIdx,
            ctx.mineral_type, outDir
        )
        
        # 5. Save results
        dataFile = os.path.join(outDir, f'{ctx.mineral_type}_Result.npz')
        
        # Clean NaN values
        Au_deep[np.isnan(Au_deep)] = 0
        F_abs[np.isnan(F_abs)] = 0
        depth_map[np.isnan(depth_map)] = 0
        f_res_MHz[np.isnan(f_res_MHz)] = 0
        moran_local[np.isnan(moran_local)] = 0
        
        final_anomaly_mask = final_mask
        inROI = ctx.inROI
        lonGrid = ctx.lonGrid
        latGrid = ctx.latGrid
        lonROI = ctx.lonROI
        latROI = ctx.latROI
        mineral_type = ctx.mineral_type
        
        # Save as .npz (Python equivalent of .mat)
        np.savez(
            dataFile,
            Au_deep=Au_deep,
            F_abs=F_abs,
            anomaly_mask_fabs=anomaly_mask_fabs,
            anomaly_mask_rededge=anomaly_mask_rededge,
            anomaly_mask_known=anomaly_mask_known,
            depth_map=depth_map,
            f_res_MHz=f_res_MHz,
            final_anomaly_mask=final_anomaly_mask,
            inROI=inROI,
            latGrid=latGrid,
            lonGrid=lonGrid,
            latROI=latROI,
            lonROI=lonROI,
            latTop=latTop,
            lonTop=lonTop,
            mineral_type=mineral_type,
            moran_local=moran_local,
            redIdx=redIdx
        )
        
        # Export KMZ
        exportKMZ(dataFile, outDir)
