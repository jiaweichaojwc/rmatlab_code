classdef IntrinsicDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            % 1. 计算原始本征吸收强度
            % (调用 GeoUtils 里的公式，这一步是对的)
            F_abs_raw = GeoUtils.computeIntrinsicAbsorption(ctx.ast, ctx.mineral_type);
            
            % 【关键步骤 1】严格复刻 both.m Line 383
            % 必须先在 ROI 内归一化到 0-1，否则后续数值量级太小
            F_abs = GeoUtils.mat2gray_roi(F_abs_raw, ctx.inROI);
            
            % 2. 计算 Moran I (这里必须用卷积 conv2，不能用通用函数！)
            % 对应 both.m Line 437-450
            
            % (A) Z-Score 标准化
            F_vals = F_abs(ctx.inROI);
            F_mean = mean(F_vals, 'omitnan');
            F_std = std(F_vals, 'omitnan');
            if F_std == 0, F_std = eps; end
            
            Z = (F_abs - F_mean) / F_std;
            Z(~ctx.inROI) = NaN;
            
            % (B) 【关键步骤 2】NaN 处理 (both.m Line 441)
            % 这里的逻辑是：把 NaN 变成 0，让它们参与卷积求和
            % (之前的通用函数是忽略 NaN，会导致边缘计算结果不同)
            Z(isnan(Z) | isinf(Z)) = 0; 
            
            % (C) 【关键步骤 3】使用 conv2 卷积求和 (both.m Line 442-443)
            kernel = ones(3,3); kernel(2,2)=0;
            local_sum = conv2(Z, kernel, 'same');
            local_sum(isnan(local_sum)) = 0;
            
            % (D) 【关键步骤 4】归一化 (both.m Line 446-448)
            % 这里的归一化是除以 max_local_sum，而不是 mat2gray(0-1)
            ls_roi = local_sum(ctx.inROI);
            max_ls = max(ls_roi(:), [], 'omitnan');
            if isempty(max_ls) || max_ls == 0, max_ls = eps; end
            
            moran_local = Z .* local_sum / max_ls;
            
            % (E) 清理无效值
            moran_local(~ctx.inROI) = NaN;
            moran_local(isnan(moran_local) | isinf(moran_local)) = 0;
            
            % 3. 动态阈值计算 (95% 分位数)
            % 对应 both.m Line 453-463
            [F_thr_base, ~, Moran_thr_base, ~] = GeoUtils.getMineralThresholds(ctx.mineral_type);
            
            % =========================================================
            % 【新增】Levashov 模式判断 (复刻 both.m Line 67-72)
            %  如果启用了微弱信号增强，阈值降低 20%
            % =========================================================
            if isprop(ctx, 'levashov_mode') && ctx.levashov_mode
                % fprintf('  [Intrinsic] 启用 Levashov 增强: 阈值 x0.8\n');
                F_thr_base = F_thr_base * 0.8;
                Moran_thr_base = Moran_thr_base * 0.8;
            end
            % =========================================================
            
            % 提取 ROI 内有效值进行统计
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
            
            % 5. 形态学降噪 (both.m Line 473)
            se = strel('square', 3);
            mask = imopen(mask, se);
            
            % 返回结果
            res.mask = mask;
            res.debug.F_abs = F_abs;
            res.debug.moran_local = moran_local;
        end
    end
end