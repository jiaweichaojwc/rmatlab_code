classdef IntrinsicDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            % 1. 计算原始本征吸收强度
            F_abs_raw = GeoUtils.computeIntrinsicAbsorption(ctx.ast, ctx.mineral_type);
            
            % 严格复刻：ROI内归一化
            F_abs = GeoUtils.mat2gray_roi(F_abs_raw, ctx.inROI);
            
            % 2. 计算 Moran I (conv2 模式)
            F_vals = F_abs(ctx.inROI);
            F_mean = mean(F_vals, 'omitnan');
            F_std = std(F_vals, 'omitnan');
            if F_std == 0, F_std = eps; end
            
            Z = (F_abs - F_mean) / F_std;
            Z(~ctx.inROI) = NaN;
            Z(isnan(Z) | isinf(Z)) = 0; 
            
            kernel = ones(3,3); kernel(2,2)=0;
            local_sum = conv2(Z, kernel, 'same');
            local_sum(isnan(local_sum)) = 0;
            
            ls_roi = local_sum(ctx.inROI);
            max_ls = max(ls_roi(:), [], 'omitnan');
            if isempty(max_ls) || max_ls == 0, max_ls = eps; end
            
            moran_local = Z .* local_sum / max_ls;
            
            moran_local(~ctx.inROI) = NaN;
            moran_local(isnan(moran_local) | isinf(moran_local)) = 0;
            
            % 3. 动态阈值计算 (核心修复部分)
            [F_thr_base, ~, Moran_thr_base, ~] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            
            % 【关键修复】检查 Levashov 模式并调整基础阈值
            % 必须与 Main.m 中的配置对应 (config.levashov_mode)
            if isprop(ctx, 'levashov_mode') && ctx.levashov_mode
                % fprintf('  [Intrinsic] 启用 Levashov 增强 (阈值 x0.8)\n');
                F_thr_base = F_thr_base * 0.8;
                Moran_thr_base = Moran_thr_base * 0.8;
            end
            
            % 提取 ROI 内有效值
            F_roi_vals = F_abs(ctx.inROI & ~isnan(F_abs));
            m_roi_vals = moran_local(ctx.inROI & ~isnan(moran_local));
            
            if isempty(F_roi_vals), F_dyn = 0; else, F_dyn = prctile(F_roi_vals, 95); end
            if isempty(m_roi_vals), M_dyn = 0; else, M_dyn = prctile(m_roi_vals, 95); end
            
            % 融合阈值
            F_final = max(F_thr_base, F_dyn * 0.9);
            M_final = max(Moran_thr_base, M_dyn * 0.9);
            
            % 4. 生成掩码
            cond_F = F_abs > F_final;
            cond_M = moran_local > M_final;
            cond_Valid = ~isnan(F_abs) & ~isinf(F_abs) & ~isnan(moran_local);
            
            mask = cond_F & cond_M & cond_Valid & ctx.inROI;
            mask = double(mask);
            
            % 5. 形态学降噪
            se = strel('square', 3);
            mask = imopen(mask, se);
            
            res.mask = mask;
            res.debug.F_abs = F_abs;
            res.debug.moran_local = moran_local;
        end
    end
end