classdef IntrinsicDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            fprintf('  [Intrinsic] 计算本征吸收连续异常面 (专属等值线模式)...\n');
            
            % 1. 计算原始本征吸收强度
            F_abs_raw = GeoUtils.computeIntrinsicAbsorption(ctx.ast, ctx.mineral_type);
            F_abs = GeoUtils.mat2gray_roi(F_abs_raw, ctx.inROI);
            
            % 2. 计算 Moran I (空间自相关)
            F_vals = F_abs(ctx.inROI);
            F_mean = mean(F_vals, 'omitnan');
            F_std = std(F_vals, 'omitnan');
            if F_std == 0, F_std = eps; end
            
            Z = (F_abs - F_mean) / F_std;
            Z(~ctx.inROI) = NaN;
            
            local_sum = GeoUtils.calc_local_sum_with_nan(Z);
            
            ls_roi = local_sum(ctx.inROI);
            max_ls = max(ls_roi(:), [], 'omitnan');
            if isempty(max_ls) || max_ls == 0, max_ls = eps; end
            
            moran_local = Z .* local_sum / max_ls;
            moran_local(~ctx.inROI) = NaN;
            moran_local(isnan(moran_local) | isinf(moran_local)) = 0;
            
            % =========================================================
            % 3. 【核心重写】生成连续热力图梯度面，抛弃硬阈值
            % =========================================================
            F_norm = GeoUtils.mat2gray_roi(F_abs, ctx.inROI);
            M_norm = GeoUtils.mat2gray_roi(moran_local, ctx.inROI);
            
            F_norm(isnan(F_norm)) = 0;
            M_norm(isnan(M_norm)) = 0;
            
            % 综合异常强度 (60%依赖吸收强度，40%依赖空间聚集度)
            continuous_mask = 0.6 * F_norm + 0.4 * M_norm;
            
            % 高斯平滑生成连片的异常晕圈 (画等值线的关键)
            continuous_mask = imgaussfilt(continuous_mask, 4, 'Padding', 'replicate');
            continuous_mask(~ctx.inROI) = 0;
            
            % 4. 封装返回
            res.mask = double(continuous_mask); % 返回连续的概率面
            res.debug.F_abs = F_abs;
            res.debug.moran_local = moran_local;
        end
    end
end