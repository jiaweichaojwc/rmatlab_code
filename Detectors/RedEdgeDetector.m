classdef RedEdgeDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            % 从 Context 获取波段
            B4 = GeoUtils.getBand(ctx.s2, 3);
            B5 = GeoUtils.getBand(ctx.s2, 7);
            B6 = GeoUtils.getBand(ctx.s2, 8);
            B7 = GeoUtils.getBand(ctx.s2, 9);
            
            % 1. 计算 S2REP
            scale_factors = [1.9997e-05, 1.9998e-05, 1.9998e-05, 1.9999e-05];
            offsets = [-0.1, -0.1, -0.1, -0.1];
            [S2REP, ~] = GeoUtils.calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets);
            
            % 2. 异常强度
            lambda_center = 705;
            delta_red_edge = S2REP - lambda_center;
            F_map = abs(delta_red_edge) ./ lambda_center;
            
            % 3. Moran I 计算
            moran_local = GeoUtils.computeMoranLocal(F_map);
            
            % 4. 阈值判断 (获取阈值)
            [F_thr, delta_thr, Moran_thr, ~] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            
            mask = (F_map > F_thr) & (delta_red_edge < delta_thr) & (moran_local > Moran_thr);
            mask(isnan(mask)) = 0;
            
            % 返回结果包
            res.mask = double(mask);
            res.debug.F_map = F_map;
            res.debug.delta_red_edge = delta_red_edge;
            res.debug.moran_local = moran_local;
        end
    end
end