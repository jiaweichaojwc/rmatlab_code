classdef RedEdgeDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            % 1. 获取波段
            B4 = GeoUtils.getBand(ctx.s2, 3);
            B5 = GeoUtils.getBand(ctx.s2, 7);
            B6 = GeoUtils.getBand(ctx.s2, 8);
            B7 = GeoUtils.getBand(ctx.s2, 9);
            
            % 2. 计算 S2REP (红边位置)
            scale_factors = [1.9997e-05, 1.9998e-05, 1.9998e-05, 1.9999e-05];
            offsets = [-0.1, -0.1, -0.1, -0.1];
            [S2REP, ~] = GeoUtils.calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets);
            
            % 3. 异常强度 F_map 计算
            lambda_center = 705;
            delta_red_edge = S2REP - lambda_center;
            F_map = abs(delta_red_edge) ./ lambda_center;
            
            % 4. Moran I 计算 (严格复刻 untitled.m 逻辑)
            
            % (A) Z-score 统计
            % untitled.m 使用的是全局统计 mean(F_map(:))，为了复刻效果，我们保持一致
            % 如果想更严谨地只统计 ROI 内，可以用注释掉的那行
            
            % --- 方案 A: 严格复刻 untitled.m (全局统计) ---
            F_mean = mean(F_map(:), 'omitnan');
            F_std = std(F_map(:), 'omitnan');
            
            % --- 方案 B: 仅 ROI 统计 (理论上更科学，但会导致结果与原版不同) ---
            % F_vals = F_map(ctx.inROI);
            % F_mean = mean(F_vals, 'omitnan');
            % F_std = std(F_vals, 'omitnan');
            
            if F_std == 0, F_std = eps; end
            Z = (F_map - F_mean) ./ F_std;
            
            % (B) Local Sum (调用 GeoUtils)
            ls = GeoUtils.calc_local_sum_with_nan(Z);
            
            % (C) 原始 Moran 值
            moran_raw = Z .* ls;
            
            % (D) 归一化 (复刻 untitled.m 的 mat2gray 逻辑)
            % 原版逻辑是对所有非 NaN 值进行归一化
            moran_local = nan(size(moran_raw));
            valid_mask = ~isnan(moran_raw); % 注意：原版是在全图范围归一化
            
            if any(valid_mask(:))
                min_v = min(moran_raw(valid_mask));
                max_v = max(moran_raw(valid_mask));
                if max_v - min_v < eps
                    moran_local(valid_mask) = 0;
                else
                    moran_local(valid_mask) = (moran_raw(valid_mask) - min_v) / (max_v - min_v);
                end
            end
            
            % 清理无效区域 (ROI 外置 0)
            moran_local(~ctx.inROI) = 0;
            moran_local(isnan(moran_local)) = 0;
            
            % 5. 阈值获取与 Levashov 修正
            [F_thr, delta_thr, Moran_thr, ~] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            
            % ==========================================================
            % 【核心修复】Levashov 模式阈值打折 (对应 untitled.m 第 120 行)
            % ==========================================================
            is_levashov = true; % 默认为 true 以匹配原脚本设定
            if isprop(ctx, 'levashov_mode')
                is_levashov = ctx.levashov_mode;
            end
            
            if is_levashov
                % fprintf('  [RedEdge] 启用 Levashov 增强: 阈值 x0.8, 负向阈值 x1.2\n');
                F_thr = F_thr * 0.8;
                Moran_thr = Moran_thr * 0.8;
                delta_thr = delta_thr * 1.2; % 负值放宽
            end
            % ==========================================================
            
            % 6. 生成掩码
            % 严格复刻筛选条件
            mask = (F_map > F_thr) & ...
                   (delta_red_edge < delta_thr) & ...
                   (moran_local > Moran_thr) & ...
                   ctx.inROI & ~isnan(F_map);
            
            mask = double(mask);
            mask(isnan(mask)) = 0;
            
            % 7. 返回结果
            res.mask = mask;
            % 将中间结果存入 debug，方便可视化检查
            res.debug.F_map = F_map;
            res.debug.delta_red_edge = delta_red_edge;
            res.debug.moran_local = moran_local;
        end
    end
end