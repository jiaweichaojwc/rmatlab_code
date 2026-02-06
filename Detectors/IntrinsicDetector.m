classdef IntrinsicDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            % 1. 计算 F_abs
            F_abs = GeoUtils.computeIntrinsicAbsorption(ctx.ast, ctx.mineral_type);
            F_abs = GeoUtils.mat2gray_roi(F_abs, ctx.inROI);
            
            % 2. Moran I
            moran_local = GeoUtils.computeMoranLocalROI(F_abs, ctx.inROI);
            
            % 3. 动态阈值逻辑 (复刻原逻辑)
            [F_thr, ~, Moran_thr, ~] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            
            F_vals = F_abs(ctx.inROI & ~isnan(F_abs));
            m_vals = moran_local(ctx.inROI & ~isnan(moran_local));
            
            F_dyn = prctile(F_vals, 95);
            M_dyn = prctile(m_vals, 95);
            
            F_final = max(F_thr, F_dyn * 0.9);
            M_final = max(Moran_thr, M_dyn * 0.9);
            
            mask = (F_abs > F_final) & (moran_local > M_final) & ctx.inROI;
            
            % 形态学
            se = strel('square', 3);
            mask = imopen(mask, se);
            
            res.mask = double(mask);
            res.debug.F_abs = F_abs;
            res.debug.moran_local = moran_local;
        end
    end
end