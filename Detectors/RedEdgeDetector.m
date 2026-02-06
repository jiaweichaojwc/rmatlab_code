classdef RedEdgeDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            % 获取波段
            B4 = GeoUtils.getBand(ctx.s2, 3);
            B5 = GeoUtils.getBand(ctx.s2, 7);
            B6 = GeoUtils.getBand(ctx.s2, 8);
            B7 = GeoUtils.getBand(ctx.s2, 9);
            
            % 1. 计算 S2REP
            scale_factors = [1.9997e-05, 1.9998e-05, 1.9998e-05, 1.9999e-05];
            offsets = [-0.1, -0.1, -0.1, -0.1];
            [S2REP, ~] = GeoUtils.calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets);
            
            % 2. 异常强度 F_map
            lambda_center = 705;
            delta_red_edge = S2REP - lambda_center;
            F_map = abs(delta_red_edge) ./ lambda_center;
            
            % 3. Moran I 计算 (严格复刻 untitled.m)
            % untitled.m 逻辑: Z-score -> local_sum -> 乘积 -> mat2gray归一化
            
            % (1) Z-score
            F_roi = F_map(ctx.inROI);
            F_mean = mean(F_roi, 'omitnan');
            F_std = std(F_roi, 'omitnan');
            if F_std == 0, F_std = eps; end
            Z = (F_map - F_mean) ./ F_std;
            Z(~ctx.inROI) = NaN;
            
            % (2) Local Sum
            ls = GeoUtils.calc_local_sum_with_nan(Z);
            
            % (3) Raw Moran
            moran_raw = Z .* ls;
            
            % (4) 【关键差异】归一化到 0-1 (untitled.m 第 470 行左右)
            moran_local = nan(size(moran_raw));
            valid_vals = moran_raw(ctx.inROI & ~isnan(moran_raw));
            if ~isempty(valid_vals)
                min_v = min(valid_vals);
                max_v = max(valid_vals);
                if max_v - min_v < eps
                    moran_local(ctx.inROI) = 0;
                else
                    moran_local(ctx.inROI) = (moran_raw(ctx.inROI) - min_v) / (max_v - min_v);
                end
            end
            moran_local(~ctx.inROI) = 0;
            moran_local(isnan(moran_local)) = 0;
            
            % 4. 阈值判断
            [F_thr, delta_thr, Moran_thr, ~] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            
            % 严格复刻多条件筛选
            mask = (F_map > F_thr) & ...
                   (delta_red_edge < delta_thr) & ...
                   (moran_local > Moran_thr) & ...
                   ctx.inROI & ~isnan(F_map);
            
            mask = double(mask);
            mask(isnan(mask)) = 0;
            
            res.mask = mask;
            res.debug.F_map = F_map;
            res.debug.delta_red_edge = delta_red_edge;
            res.debug.moran_local = moran_local;
        end
    end
end