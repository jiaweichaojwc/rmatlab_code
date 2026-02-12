classdef PostProcessor
    methods (Static)
        function run(ctx, engine, final_mask, outDir)
            fprintf('=== 进入后处理阶段 (多掩码增强版) ===\n');
            
            function res = safeGet(name)
                if engine.results.isKey(name)
                    res = engine.getResult(name);
                else
                    res.mask = zeros(size(ctx.inROI));
                    res.debug = struct();
                    res.debug.F_map = zeros(size(ctx.inROI));
                    res.debug.delta_red_edge = zeros(size(ctx.inROI));
                    res.debug.moran_local = zeros(size(ctx.inROI));
                    res.debug.F_abs = zeros(size(ctx.inROI));
                end
            end

            res_Red = safeGet('RedEdge');
            res_Int = safeGet('Intrinsic');
            res_Slow = safeGet('SlowVars');
            res_Known = safeGet('KnownAnomaly'); 
            
            anomaly_mask_rededge = res_Red.mask; 
            anomaly_mask_fabs = res_Int.mask;
            anomaly_mask_slow = res_Slow.mask;
            anomaly_mask_known = res_Known.mask; 
            
            % Debug data
            F_map = res_Red.debug.F_map;
            delta_red = res_Red.debug.delta_red_edge;
            moran_local = res_Int.debug.moran_local;
            F_abs = res_Int.debug.F_abs;             

            % 1. 深度与压力
            params = GeoUtils.getYakymchukParams(ctx.mineral_type);
            c = 3e8; epsilon_r = 16;
            f_res_MHz = params.a + params.b * exp(-params.c * abs(F_map));
            f_res_MHz(isnan(f_res_MHz)) = params.a; f_res_MHz(f_res_MHz < 0) = params.a; f_res_MHz(~ctx.inROI) = NaN;
            depth_map = c ./ (2 * f_res_MHz * 1e6 * sqrt(epsilon_r)) / 1000;
            depth_map = min(max(depth_map, 0), 4); depth_map(~ctx.inROI) = NaN;
            grad_P = 25 + 5 * depth_map; grad_P = min(max(grad_P, 0), 40); grad_P(~ctx.inROI) = NaN;
            
            % 2. 地表潜力通用变量
            [~,~,~, enh_func] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            eps_val = 1e-6;
            [H, W, ~] = size(ctx.ast);
            
            Ferric = GeoUtils.mat2gray_roi(ctx.ast(:,:,2) ./ (ctx.ast(:,:,1) + eps_val), ctx.inROI);
            Clay = GeoUtils.mat2gray_roi(ctx.ast(:,:,6) ./ (ctx.ast(:,:,7) + eps_val), ctx.inROI);
            NDVI_inv = GeoUtils.mat2gray_roi(1 - (ctx.NIR - ctx.Red) ./ (ctx.NIR + ctx.Red + eps_val), ctx.inROI);
            
            pcaInput = cat(3, ctx.ast(:,:,4:7)); pcaInput = reshape(pcaInput, H*W, 4);
            pcaInput = double(pcaInput - mean(pcaInput, 'omitnan')) ./ std(pcaInput, 'omitnan'); pcaInput(isnan(pcaInput)) = 0;
            [~, score] = pca(pcaInput); pcaResult = reshape(score, H, W, 4);
            Hydroxy_anomaly = GeoUtils.mat2gray_roi(pcaResult(:,:,2), ctx.inROI);
            Fe_anomaly = GeoUtils.mat2gray_roi(pcaResult(:,:,3), ctx.inROI);
            

            if strcmpi(ctx.mineral_type, 'cave')
                demIndices = GeoUtils.computeDEMIndices(ctx.dem, 'cave', H, W, ctx.inROI);
                Au_surface = enh_func(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv, demIndices.slope, demIndices.neg_curvature);
                Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI);
                
            elseif strcmpi(ctx.mineral_type, 'offshore_petroleum')
                OSI = GeoUtils.mat2gray_roi((ctx.Blue + ctx.Green + ctx.Red) ./ (ctx.NIR + eps_val), ctx.inROI);

                if isprop(ctx, 'SAR_dark_spot') && ~isempty(ctx.SAR_dark_spot)
                    SDS = ctx.SAR_dark_spot;
                else
                    SDS = zeros(H, W, 'single');
                end
                Au_surface = enh_func(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv, OSI, SDS);
                Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI);
                
            else
                Au_surface = enh_func(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv);
            end

            % Filter 1
            valid_mask = ctx.inROI & ~isnan(Au_surface); Au_temp = Au_surface; Au_temp(~valid_mask) = 0;
            Au_filt = imgaussfilt(Au_temp, 8, 'Padding', 'replicate');
            Au_surface(valid_mask) = Au_filt(valid_mask);
            Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI);
            
            % 融合权重
            if ~isequal(size(final_mask), size(Au_surface))
                final_mask = imresize(final_mask, size(Au_surface), 'nearest');
            end
            Au_surface(ctx.inROI) = Au_surface(ctx.inROI) .* (1 + final_mask(ctx.inROI) * 0.4);
            Au_surface(ctx.inROI & (isnan(Au_surface) | isinf(Au_surface))) = 0;
            
            % Filter 2
            valid_mask = ctx.inROI & ~isnan(Au_surface); Au_temp = Au_surface; Au_temp(~valid_mask) = 0;
            Au_filt = imgaussfilt(Au_temp, 6, 'Padding', 'replicate');
            Au_surface(valid_mask) = Au_filt(valid_mask);
            Au_deep = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI);

            % 3. Top 20
            temp = Au_deep; temp(~ctx.inROI) = 0;
            [~, idx] = sort(temp(:), 'descend'); top20 = idx(1:min(20, length(idx)));
            [topY, topX] = ind2sub([H, W], top20);
            latGrid_corrected = flipud(ctx.latGrid);
            lonTop = ctx.lonGrid(top20); latTop = latGrid_corrected(top20); redIdx = 1:length(top20);
            
            % 4. Visualization
            img_rgb = cat(3, GeoUtils.mat2gray_roi(ctx.Red, ctx.inROI), ...
                             GeoUtils.mat2gray_roi(ctx.Green, ctx.inROI), ...
                             GeoUtils.mat2gray_roi(ctx.Blue, ctx.inROI));
            img_rgb(isnan(img_rgb)) = 0;
            
            Visualizer.run_resonance(F_map, delta_red, moran_local, final_mask, ...
                depth_map*1000, grad_P, f_res_MHz, img_rgb, outDir, ctx.lonGrid, ctx.latGrid);
            
            masks_pack = {anomaly_mask_rededge, anomaly_mask_fabs, anomaly_mask_slow, anomaly_mask_known, final_mask};
            titles_pack = {'1.红边异常', '2.本征吸收', '3.慢变量突变', '4.已知KML异常', '5.集成并集'};
            Visualizer.run_mask_fusion(masks_pack, titles_pack, ctx.lonGrid, ctx.latGrid, outDir);
            
            Visualizer.run_deep_prediction(Au_deep, ctx.lonGrid, ctx.latGrid, ...
                ctx.lonROI, ctx.latROI, lonTop, latTop, redIdx, ctx.mineral_type, outDir);
            
            % 5. Save (修复：增加 anomaly_mask_slow 的保存)
            dataFile = fullfile(outDir, sprintf('%s_Result.mat', ctx.mineral_type));
            Au_deep(isnan(Au_deep)) = 0; F_abs(isnan(F_abs)) = 0; depth_map(isnan(depth_map)) = 0;
            f_res_MHz(isnan(f_res_MHz)) = 0; moran_local(isnan(moran_local)) = 0;
            
            final_anomaly_mask = final_mask; inROI = ctx.inROI;
            lonGrid = ctx.lonGrid; latGrid = ctx.latGrid; lonROI = ctx.lonROI; latROI = ctx.latROI;
            mineral_type = ctx.mineral_type;
            
            % 👇 把 anomaly_mask_slow 加进去了
            save(dataFile, 'Au_deep', 'F_abs', 'anomaly_mask_fabs', 'anomaly_mask_rededge', ...
                'anomaly_mask_slow', 'anomaly_mask_known', ... 
                'depth_map', 'f_res_MHz', 'final_anomaly_mask', 'inROI', ...
                'latGrid', 'lonGrid', 'latROI', 'lonROI', 'latTop', 'lonTop', ...
                'mineral_type', 'moran_local', 'redIdx');
            
            exportKMZ(dataFile, outDir);
        end
    end
end