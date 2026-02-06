classdef PostProcessor
    methods (Static)
        function run(ctx, engine, final_mask, outDir)
            fprintf('=== 进入后处理阶段 ===\n');
            
            %% 1. 数据准备 (安全获取模式)
            % 这里的逻辑保持不变，用于支持灵活融合
            function res = safeGet(name)
                if engine.results.isKey(name)
                    res = engine.getResult(name);
                else
                    fprintf('  [提示] 未检测到 %s 结果，使用空掩码填充。\n', name);
                    res.mask = zeros(size(ctx.inROI));
                    res.debug.F_map = zeros(size(ctx.inROI));
                    res.debug.delta_red_edge = zeros(size(ctx.inROI));
                    res.debug.moran_local = zeros(size(ctx.inROI));
                    res.debug.F_abs = zeros(size(ctx.inROI));
                end
            end

            res_Red = safeGet('RedEdge');
            res_Int = safeGet('Intrinsic');
            
            % 提取绘图所需的中间变量
            F_map = res_Red.debug.F_map;
            delta_red = res_Red.debug.delta_red_edge;
            anomaly_mask_rededge = res_Red.mask; 
            
            F_abs = res_Int.debug.F_abs;             
            anomaly_mask_fabs = res_Int.mask;        
            moran_local = res_Int.debug.moran_local; 
            
            %% 2. 深度与压力计算 (保持一致)
            fprintf('  计算共振深度与压力...\n');
            params = GeoUtils.getYakymchukParams(ctx.mineral_type);
            c = 3e8; epsilon_r = 16;
            
            % 防止 F_map 全0导致计算异常
            calc_F = F_map; 
            f_res_MHz = params.a + params.b * exp(-params.c * abs(calc_F));
            f_res_MHz(isnan(f_res_MHz)) = params.a;
            f_res_MHz(~ctx.inROI) = NaN;
            
            depth_map = c ./ (2 * f_res_MHz * 1e6 * sqrt(epsilon_r)) / 1000;
            depth_map = min(max(depth_map, 0), 4);
            depth_map(~ctx.inROI) = NaN;
            
            grad_P = 25 + 5 * depth_map;
            grad_P(~ctx.inROI) = NaN;
            
            %% 3. 地表潜力计算 (核心修复：恢复 PCA 逻辑)
            fprintf('  计算地表潜力指数 (含PCA)...\n');
            [~,~,~, enh_func] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            
            % (1) 基础指数计算 (完全对齐 both.m)
            eps_val = 1e-6;
            Ferric = ctx.ast(:,:,2) ./ (ctx.ast(:,:,1) + eps_val);
            Clay   = ctx.ast(:,:,6) ./ (ctx.ast(:,:,7) + eps_val);
            NDVI   = (ctx.NIR - ctx.Red) ./ (ctx.NIR + ctx.Red + eps_val);
            NDVI_inv = 1 - NDVI;
            
            % (2) 归一化 (both.m 使用 mat2gray_roi)
            Ferric = GeoUtils.mat2gray_roi(Ferric, ctx.inROI);
            Clay = GeoUtils.mat2gray_roi(Clay, ctx.inROI);
            NDVI_inv = GeoUtils.mat2gray_roi(NDVI_inv, ctx.inROI);
            
            % (3) PCA 计算 (核心差异点！！！)
            % both.m 逻辑: 取 ASTER 4,5,6,7 波段进行 PCA
            fprintf('    执行 PCA 提取羟基与铁异常...\n');
            pcaInput = cat(3, ctx.ast(:,:,4), ctx.ast(:,:,5), ctx.ast(:,:,6), ctx.ast(:,:,7));
            [H, W, ~] = size(pcaInput);
            pcaInput = reshape(pcaInput, H*W, 4);
            
            % 标准化 (排除 NaN)
            pcaInput = double(pcaInput);
            mean_vec = mean(pcaInput, 'omitnan');
            std_vec = std(pcaInput, 'omitnan');
            pcaInput = (pcaInput - mean_vec) ./ std_vec;
            
            % 处理 NaN 以便 PCA 运行 (pca 函数不支持 NaN)
            nan_mask = any(isnan(pcaInput), 2);
            pcaInput(nan_mask, :) = 0; 
            
            [~, score] = pca(pcaInput);
            pcaResult = reshape(score, H, W, 4);
            
            % 提取主成分并归一化
            Hydroxy_anomaly = GeoUtils.mat2gray_roi(pcaResult(:,:,2), ctx.inROI); % PC2
            Fe_anomaly      = GeoUtils.mat2gray_roi(pcaResult(:,:,3), ctx.inROI); % PC3
            
            % (4) 代入公式计算 Au_surface
            % 现在传入的是真正的 PCA 结果，而不是替代品
            Au_surface = enh_func(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv); 
            Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI);
            
            % 融合异常区 (Mask 增强)
            Au_surface(ctx.inROI) = Au_surface(ctx.inROI) .* (1 + final_mask(ctx.inROI) * 0.4);
            
            % 高斯滤波 (Both.m 逻辑: sigma=8, 仅有效值参与)
            valid_mask = ctx.inROI & ~isnan(Au_surface);
            Au_temp = Au_surface; 
            Au_temp(~valid_mask) = 0;
            
            % 使用 imgaussfilt (sigma=8, padding=replicate 对应 both.m)
            Au_filt = imgaussfilt(Au_temp, 8, 'Padding', 'replicate');
            
            Au_surface(valid_mask) = Au_filt(valid_mask);
            Au_deep = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI);
            
            %% 4. 地质约束
            fprintf('  应用地质约束...\n');
            temp = Au_deep; temp(~ctx.inROI) = 0;
            [~, idx] = sort(temp(:), 'descend');
            top20 = idx(1:min(20, length(idx)));
            [topY, topX] = ind2sub([H, W], top20);
            
            latGrid_corrected = flipud(ctx.latGrid); 
            lonTop = ctx.lonGrid(top20);
            latTop = latGrid_corrected(top20);
            redIdx = 1:length(top20); 
            
            %% 5. 可视化
            fprintf('>>> 开始绘图...\n');
            
            % 准备 RGB
            img_rgb = cat(3, GeoUtils.mat2gray_roi(ctx.Red, ctx.inROI), ...
                             GeoUtils.mat2gray_roi(ctx.Green, ctx.inROI), ...
                             GeoUtils.mat2gray_roi(ctx.Blue, ctx.inROI));
            img_rgb(isnan(img_rgb)) = 0;

            Visualizer.run_resonance(F_map, delta_red, moran_local, final_mask, ...
                depth_map*1000, grad_P, f_res_MHz, img_rgb, outDir, ctx.lonGrid, ctx.latGrid);
            
            red_dilated = imdilate(anomaly_mask_rededge, strel('disk', 2));
            
            % 获取慢变量掩码(如果有)
            res_Slow = safeGet('SlowVars');
            anomaly_mask_slow = res_Slow.mask;

            Visualizer.run_mask_fusion(red_dilated, anomaly_mask_fabs, anomaly_mask_slow, ...
                final_mask, ctx.lonGrid, ctx.latGrid, outDir);
                
            Visualizer.run_deep_prediction(Au_deep, ctx.lonGrid, ctx.latGrid, ...
                ctx.lonROI, ctx.latROI, lonTop, latTop, redIdx, ctx.mineral_type, outDir);

            %% 6. 保存数据
            dataFile = fullfile(outDir, sprintf('%s_Result.mat', ctx.mineral_type));
            
            final_anomaly_mask = final_mask;
            inROI = ctx.inROI;
            lonGrid = ctx.lonGrid;
            latGrid = ctx.latGrid;
            lonROI = ctx.lonROI;
            latROI = ctx.latROI;
            mineral_type = ctx.mineral_type;
            
            Au_deep(isnan(Au_deep)) = 0;
            F_abs(isnan(F_abs)) = 0;
            depth_map(isnan(depth_map)) = 0;
            f_res_MHz(isnan(f_res_MHz)) = 0;
            moran_local(isnan(moran_local)) = 0;
            
            save(dataFile, 'Au_deep', 'F_abs', 'anomaly_mask_fabs', 'anomaly_mask_rededge', ...
                'depth_map', 'f_res_MHz', 'final_anomaly_mask', 'inROI', ...
                'latGrid', 'lonGrid', 'latROI', 'lonROI', 'latTop', 'lonTop', ...
                'mineral_type', 'moran_local', 'redIdx');
            
            fprintf('  数据已保存至: %s\n', dataFile);
            
            %% 7. 导出 KMZ
            exportKMZ(dataFile, outDir);
        end
    end
end