classdef PostProcessor
    methods (Static)
        function run(ctx, engine, final_mask, outDir)
            fprintf('=== 进入后处理阶段 ===\n');
            
            %% 1. 数据准备 (Data Preparation)
            img_rgb = cat(3, GeoUtils.mat2gray_roi(ctx.Red, ctx.inROI), ...
                             GeoUtils.mat2gray_roi(ctx.Green, ctx.inROI), ...
                             GeoUtils.mat2gray_roi(ctx.Blue, ctx.inROI));
            img_rgb(isnan(img_rgb)) = 0;
            
            % 获取各探测器的详细结果
            res_Red = engine.getResult('RedEdge');
            res_Int = engine.getResult('Intrinsic');
            res_Slow = engine.getResult('SlowVars');
            
            % --- 恢复原始变量名 ---
            % 1. 红边相关
            F_map = res_Red.debug.F_map;
            delta_red = res_Red.debug.delta_red_edge;
            anomaly_mask_rededge = res_Red.mask; % [截图变量]
            
            % 2. 本征吸收相关
            F_abs = res_Int.debug.F_abs;             % [截图变量]
            anomaly_mask_fabs = res_Int.mask;        % [截图变量]
            moran_local = res_Int.debug.moran_local; % [截图变量] (保留最后计算的本征Moran)
            
            %% 2. 深度与压力计算
            fprintf('  计算共振深度与压力...\n');
            params = GeoUtils.getYakymchukParams(ctx.mineral_type);
            c = 3e8; epsilon_r = 16;
            
            f_res_MHz = params.a + params.b * exp(-params.c * abs(F_map));
            f_res_MHz(isnan(f_res_MHz)) = params.a;
            f_res_MHz(f_res_MHz < 0) = params.a;
            f_res_MHz(~ctx.inROI) = NaN;
            
            % 计算深度 (km) - 对应截图中的 depth_map
            depth_map = c ./ (2 * f_res_MHz * 1e6 * sqrt(epsilon_r)) / 1000;
            depth_map = min(max(depth_map, 0), 4);
            depth_map(~ctx.inROI) = NaN;
            
            % 为了绘图方便，也保留米为单位的变量
            depth_m = depth_map * 1000;
            
            grad_P = 25 + 5 * depth_map;
            grad_P(~ctx.inROI) = NaN;
            
            %% 3. 地表潜力计算
            fprintf('  计算地表潜力指数...\n');
            [~,~,~, enh_func] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            
            Ferric = GeoUtils.mat2gray_roi(ctx.ast(:,:,2)./(ctx.ast(:,:,1)+eps), ctx.inROI);
            Clay   = GeoUtils.mat2gray_roi(ctx.ast(:,:,6)./(ctx.ast(:,:,7)+eps), ctx.inROI);
            NDVI   = (ctx.NIR - ctx.Red)./(ctx.NIR + ctx.Red + eps);
            NDVI_inv = GeoUtils.mat2gray_roi(1-NDVI, ctx.inROI);
            
            % 计算 Au_surface
            % 注意：此处演示使用通用参数，实际会自动匹配矿种公式
            Au_surface = enh_func(Ferric, Ferric, Clay, Clay, NDVI_inv); 
            Au_surface = GeoUtils.mat2gray_roi(Au_surface, ctx.inROI);
            
            % 融合异常区
            Au_surface(ctx.inROI) = Au_surface(ctx.inROI) .* (1 + final_mask(ctx.inROI) * 0.4);
            
            % 高斯滤波
            valid_mask = ctx.inROI & ~isnan(Au_surface);
            Au_temp = Au_surface; Au_temp(~valid_mask) = 0;
            Au_filt = imgaussfilt(Au_temp, 6, 'Padding', 'replicate');
            Au_deep = Au_surface; 
            Au_deep(valid_mask) = Au_filt(valid_mask);
            Au_deep = GeoUtils.mat2gray_roi(Au_deep, ctx.inROI);
            
            %% 4. 地质约束与 Top 点提取
            fprintf('  应用地质约束...\n');
            [H, W] = size(Au_deep);
            temp = Au_deep; temp(~ctx.inROI) = 0;
            [~, idx] = sort(temp(:), 'descend');
            top20 = idx(1:min(20, length(idx)));
            [topY, topX] = ind2sub([H, W], top20);
            
            latGrid_corrected = flipud(ctx.latGrid); 
            lonTop = ctx.lonGrid(top20);
            latTop = latGrid_corrected(top20);
            redIdx = 1:length(top20); % 默认全通过，实际可加逻辑
            
            %% 5. 可视化
            fprintf('>>> 开始绘图...\n');
            Visualizer.run_resonance(F_map, delta_red, moran_local, final_mask, ...
                depth_m, grad_P, f_res_MHz, img_rgb, outDir, ctx.lonGrid, ctx.latGrid);
            
            red_dilated = imdilate(res_Red.mask, strel('disk', 2));
            Visualizer.run_mask_fusion(red_dilated, anomaly_mask_fabs, res_Slow.mask, ...
                final_mask, ctx.lonGrid, ctx.latGrid, outDir);
                
            Visualizer.run_deep_prediction(Au_deep, ctx.lonGrid, ctx.latGrid, ...
                ctx.lonROI, ctx.latROI, lonTop, latTop, redIdx, ctx.mineral_type, outDir);

            %% 6. 保存数据 (完全匹配截图格式)
            dataFile = fullfile(outDir, sprintf('%s_Result.mat', ctx.mineral_type));
            
            % 准备保存的变量 (与截图一一对应)
            final_anomaly_mask = final_mask; % 重命名以匹配
            inROI = ctx.inROI;
            lonGrid = ctx.lonGrid;
            latGrid = ctx.latGrid;
            lonROI = ctx.lonROI;
            latROI = ctx.latROI;
            mineral_type = ctx.mineral_type;
            
            % 将 NaN 替换为 0 (保持原脚本习惯)
            Au_deep(isnan(Au_deep)) = 0;
            F_abs(isnan(F_abs)) = 0;
            depth_map(isnan(depth_map)) = 0;
            f_res_MHz(isnan(f_res_MHz)) = 0;
            moran_local(isnan(moran_local)) = 0;
            
            % 保存所有变量
            save(dataFile, ...
                'Au_deep', ...
                'F_abs', ...
                'anomaly_mask_fabs', ...
                'anomaly_mask_rededge', ...
                'depth_map', ...        % 注意：这是 km 单位
                'f_res_MHz', ...
                'final_anomaly_mask', ...
                'inROI', ...
                'latGrid', 'lonGrid', ...
                'latROI', 'lonROI', ...
                'latTop', 'lonTop', ...
                'mineral_type', ...
                'moran_local', ...
                'redIdx');
            
            fprintf('  数据已保存至: %s\n', dataFile);
            
            %% 7. 导出 KMZ
            % 这里的 dataFile 已经包含了 lonGrid，Python 脚本不会再报错了
            exportKMZ(dataFile, outDir);
        end
    end
end