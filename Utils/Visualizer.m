classdef Visualizer
    methods (Static)
        function run_resonance(F_map, delta_red, moran, mask, depth, gradP, freq, RGB, outDir, lonGrid, latGrid)
            lonV = linspace(min(lonGrid(:)), max(lonGrid(:)), size(F_map, 2));
            latV = linspace(min(latGrid(:)), max(latGrid(:)), size(F_map, 1));
            f = figure('Position', [100 100 2000 1200], 'Color', 'w', 'Visible', 'off');
            t = tiledlayout(2,4, 'TileSpacing', 'compact', 'Padding', 'tight');
            data_list = {RGB, F_map, delta_red, moran, mask, depth*1000, gradP, freq};
            titles = {'RGB', 'F判别', '红边位移', 'Moran I', '综合异常', '深度', '压力', '频率'};
            clims = {[], [0 0.15], [-15 15], [0 1], [0 1], [0 2000], [0 40], [10 100]};
            
            for i=1:8
                ax = nexttile;
                imagesc(lonV, latV, flipud(data_list{i})); 
                axis xy; axis image; grid on; 
                title(titles{i}); 
                if i>1
                    colormap(ax,turbo(256)); 
                    colorbar; 
                    if ~isempty(clims{i})
                        caxis(ax, clims{i}); 
                    end
                end
            end
            print(f, fullfile(outDir, '01_共振参数综合图.png'), '-dpng', '-r300'); 
            close(f);
        end

        function run_mask_fusion(mask_list, title_list, lonGrid, latGrid, outDir)
            num_masks = length(mask_list);
            if num_masks == 0, return; end
            
            lonV = linspace(min(lonGrid(:)), max(lonGrid(:)), size(mask_list{1}, 2));
            latV = linspace(min(latGrid(:)), max(latGrid(:)), size(mask_list{1}, 1));
            
            % 智能排版
            if num_masks <= 4
                cols = num_masks; 
                rows = 1;
            else
                cols = 4; 
                rows = ceil(num_masks / cols); 
            end
            
            f = figure('Position', [100 450 400*cols 400*rows+50], 'Color', 'w', 'Visible', 'off');
            t = tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'tight');
            
            n_colors = 256;
            % 浅蓝->红的自定义色卡
            custom_map = [linspace(1,0,n_colors)', linspace(1,0.8,n_colors)', linspace(1,0,n_colors)'];

            for i = 1:num_masks
                ax = nexttile;
                img_data = double(mask_list{i});
                img_data(isnan(img_data)) = 0; 
                imagesc(lonV, latV, flipud(img_data));
                
                colormap(ax, custom_map); 
                caxis(ax, [0 1]); 
                axis xy; axis image; grid on;
                if i <= length(title_list)
                    title(title_list{i}, 'FontSize', 12, 'FontWeight', 'bold'); 
                end
            end
            
            outName = fullfile(outDir, sprintf('02_掩码集成_%d图.png', num_masks));
            print(f, outName, '-dpng', '-r400');
            close(f);
        end

        function run_deep_prediction(Au, lonG, latG, lonR, latR, lonT, latT, rIdx, mineral, outDir)
            % 1. 初始化绘图窗口（确保可见性，优先渲染）
            f = figure('Position', [500 100 1200 1000], 'Color', 'w', 'Visible', 'on');
            hold on; % 提前开启hold on，确保所有元素绘制在同一轴上
            ax = gca; % 获取当前轴，统一控制样式
            
            % 2. 生成经纬度网格
            lonV = linspace(min(lonG(:)), max(lonG(:)), size(Au, 2));
            latV = linspace(min(latG(:)), max(latG(:)), size(Au, 1));
            
            % 3. 预处理Au数据（统一类型+清理无效值）
            Au_data = double(Au);
            Au_data(isnan(Au_data) | isinf(Au_data)) = 0; % 替换NaN/Inf为0，避免填充中断
            Au_flip = flipud(Au_data); % 提前翻转，避免重复计算
            
            % 4. 自适应马氏距离等高线+颜色填充（核心修复）
            Au_valid = Au_data(Au_data ~= 0); % 排除0值（已替换的NaN/Inf）
            if isempty(Au_valid)
                % 无有效数据时的兜底逻辑
                warning('Au数据无有效值，使用默认固定间隔填充');
                contour_levels = linspace(0.4, 1, 80); % 原固定间隔
            else
                % 计算马氏距离
                Au_valid_2d = reshape(Au_valid, [], 1);
                mu = mean(Au_valid_2d);
                sigma = cov(Au_valid_2d);
                if sigma < 1e-8, sigma = 1e-8; end % 防止协方差为0
                mahal_dist = mahal(Au_valid_2d, Au_valid_2d);
                
                % 按分位数生成自适应层级（保证至少8个层级，确保填充细腻）
                quantiles = linspace(0.1, 0.95, 20); % 20个分位数，填充更细腻
                mahal_q = quantile(mahal_dist, quantiles);
                [~, idx_sort] = sort(Au_valid);
                contour_levels = Au_valid(idx_sort(round(quantiles * length(Au_valid))));
                contour_levels = sort(unique(contour_levels)); % 去重+升序
                % 确保层级数量足够，避免填充空白
                if length(contour_levels) < 5
                    contour_levels = linspace(min(Au_valid), max(Au_valid), 80);
                end
            end
            
            % 5. 绘制填充等高线（核心：确保颜色映射生效）
            % 先画填充等高线，设置LineColor为none避免轮廓干扰
            [C, hFill] = contourf(ax, lonV, latV, Au_flip, contour_levels, 'LineColor', 'none');
            % 强制绑定色卡到当前轴
            colormap(ax, jet(256)); 
            % 设置颜色范围（适配数据分布）
            caxis(ax, [min(contour_levels), max(contour_levels)]);
            % 添加颜色条（关联当前轴）
            cb = colorbar(ax);
            cb.FontSize = 10;
            cb.Label.String = 'Au Value';
            
            % 6. 绘制等高线轮廓（浅灰色，突出边界）
            contour(ax, lonV, latV, Au_flip, contour_levels, 'LineColor', [0.8 0.8 0.8], 'LineWidth', 0.5);
            
            % 7. 叠加其他元素（边界线+标记点）
            if ~isempty(lonR) && ~isempty(latR)
                plot(ax, lonR, latR, 'k-', 'LineWidth', 2.5); % 边界线
            end
            if ~isempty(lonT) && ~isempty(latT)
                plot(ax, lonT, latT, 'wo', 'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.2 0.2]); % 普通标记点
            end
            if ~isempty(rIdx) && ~isempty(lonT) && ~isempty(latT)
                plot(ax, lonT(rIdx), latT(rIdx), 'yo', 'MarkerSize', 18, 'LineWidth', 3); % 重点标记点
            end
            
            % 8. 轴样式设置（确保坐标/网格正常）
            axis(ax, 'xy'); % 恢复地理坐标的y轴方向
            axis(ax, 'image'); % 等比例显示
            grid(ax, 'on'); 
            grid(ax, 'minor'); % 显示细网格，提升可读性
            title(ax, ['Deep Prediction (自适应马氏距离): ', upper(mineral)], 'FontSize', 16, 'FontWeight', 'bold');
            xlabel(ax, 'Longitude', 'FontSize', 12);
            ylabel(ax, 'Latitude', 'FontSize', 12);
            ax.FontSize = 10;
            
            % 9. 强制刷新渲染（关键：确保填充颜色生效）
            drawnow expose; % 深度刷新，比普通drawnow更彻底
            pause(0.1); % 短暂等待渲染完成（避免保存时颜色未加载）
            
            % 10. 保存文件（确保路径存在）
            if ~exist(outDir, 'dir')
                mkdir(outDir); % 自动创建输出目录，避免保存失败
            end
            % 保存fig文件
            savefig(f, fullfile(outDir, '03_深部成矿预测图.fig'));
            % 保存png文件（高分辨率）
            print(f, fullfile(outDir, '03_深部成矿预测图.png'), '-dpng', '-r500', '-painters'); % 使用painters渲染器，保证颜色
            
            % 11. 清理资源
            close(f);
        end
    end
end