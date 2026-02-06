classdef SlowVarsDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            inROI = ctx.inROI;
            
            % 1. 地应力
            [gx, gy] = gradient(ctx.dem);
            stress_grad = sqrt(gx.^2 + gy.^2);
            
            % 2. 氧化还原
            iron_oxide = ctx.lan(:,:,3) ./ (ctx.lan(:,:,2) + eps);
            swir1 = ctx.s2(:,:,8); swir2 = ctx.s2(:,:,9);
            oxy_fug = swir2 - mean(cat(3, swir1, swir2), 3);
            oxy_fug(isnan(oxy_fug)|isinf(oxy_fug)) = 0;
            redox_grad = abs(iron_oxide - mean(iron_oxide(inROI),'omitnan')) + ...
                         abs(oxy_fug - mean(oxy_fug(inROI),'omitnan'));
            
            % 3. 流体超压
            tir_mean = mean(ctx.ast(:,:,10:14), 3);
            ndvi = (ctx.NIR - ctx.Red) ./ (ctx.NIR + ctx.Red + eps);
            fluid_over = tir_mean + 3*(1 - ndvi);
            
            % 4. 断裂
            edges = edge(stress_grad, 'canny', [0.05 0.25]);
            fault_activity = bwareaopen(edges, 50) .* stress_grad;
            
            % 5. 盖层
            carbonate = (ctx.ast(:,:,6) + ctx.ast(:,:,8)) ./ (ctx.ast(:,:,7) + eps);
            
            % 6. 温度梯度
            [gtx, gty] = gradient(tir_mean);
            temp_grad = sqrt(gtx.^2 + gty.^2);
            
            % 7. 化学势
            [gcx, gcy] = gradient(iron_oxide + oxy_fug);
            chem_grad = sqrt(gcx.^2 + gcy.^2);
            
            % Z-Score helper
            z = @(x) (x - mean(x(inROI),'omitnan')) ./ (std(x(inROI),'omitnan') + eps);
            
            % 组合
            a = -(0.5 * z(carbonate) + 0.5 * z(temp_grad));
            b = 0.25*z(stress_grad) + 0.2*z(redox_grad) + 0.25*z(fluid_over) + ...
                0.15*z(fault_activity) + 0.15*z(chem_grad);
            
            Delta = b.^2 + (8/27) * a.^3;
            mask = (Delta < 0) & inROI;
            mask = bwareaopen(mask, 100);
            mask = imdilate(mask, strel('disk', 8));
            
            res.mask = double(mask);
            res.debug.Delta = Delta;
        end
    end
end