classdef Visualizer
    methods (Static)
        %% ================= 1. 共振参数综合图 (viz_resonance) =================
        function run_resonance(F_map, delta_red, moran, mask, depth, gradP, freq, RGB, outDir, lonGrid, latGrid)
            % 1. 建立升序坐标向量
            lonV = linspace(min(lonGrid(:)), max(lonGrid(:)), size(F_map, 2));
            latV = linspace(min(latGrid(:)), max(latGrid(:)), size(F_map, 1));
        
            f = figure('Position', [100 100 2000 1200], 'Color', 'w', 'Visible', 'off'); % 暂时不可见，画完再存
            t = tiledlayout(2,4, 'TileSpacing', 'compact', 'Padding', 'tight');
        
            % 数据列表
            data_list = {RGB, F_map, delta_red, moran, mask, depth, gradP, freq};
            titles = {'RGB 真实色彩', 'F(λ) 判别函数', '红边位移 Δλ (nm)', '局部 Morans I', ...
                      '综合异常靶区', '估算深度 H (m)', '压力梯度 (MPa/km)', '共振频率 (MHz)'};
            
            % 色阶显示范围
            clims = {[], [0 0.15], [-15 15], [0 1], [0 1], [0 2000], [0 40], [10 100]};
            
            % 刻度
            ticks_list = {[], 0:0.02:0.15, -15:3:15, 0:0.1:1, 0:0.1:1, 0:100:2000, 0:5:40, 10:10:100};
        
            for i = 1:8
                ax = nexttile;
                % ⚠️ 注意：这里数据是否翻转取决于传入的数据方向
                % 在 OOP 架构中，如果数据已经是翻转好的，这里可能需要调整
                % 为了保持原代码逻辑，这里保留 flipud，如果图倒了，去掉 flipud 即可
                imagesc(lonV, latV, flipud(data_list{i}));
                
                axis xy; axis image; grid on;            
                title(titles{i}, 'FontSize', 12, 'FontWeight', 'bold');
                
                xticks(linspace(min(lonV), max(lonV), 6));
                yticks(linspace(min(latV), max(latV), 6));
                xtickformat('%.3f°'); ytickformat('%.3f°');
                ax.FontSize = 9;
        
                if ~isempty(clims{i}), caxis(ax, clims{i}); end
                
                if i ~= 1
                    colormap(ax, turbo(256)); 
                    cb = colorbar;
                    if ~isempty(ticks_list{i}), set(cb, 'Ticks', ticks_list{i}); end
                end
            end
            
            % 导出
            outFile = fullfile(outDir, '01_共振参数综合图_蓝红精细版.png');
            fprintf('  Saving: %s\n', outFile);
            print(f, outFile, '-dpng', '-r300');
            close(f);
        end

        %% ================= 2. 掩码集成对比图 (viz_mask_fusion) =================
        function run_mask_fusion(re, fa, ss, final, lonGrid, latGrid, outDir)
            f = figure('Position', [100 450 1600 500], 'Color', 'w', 'Visible', 'off');
            t = tiledlayout(1, 4, 'TileSpacing', 'compact', 'Padding', 'tight');
            
            lonV = linspace(min(lonGrid(:)), max(lonGrid(:)), size(re, 2));
            latV = linspace(min(latGrid(:)), max(latGrid(:)), size(re, 1));
            
            titles = {'红边异常(加粗)', '本征吸收异常', '慢变量(尖点突变)', '集成总靶区'};
            data = {re, fa, ss, final};
        
            % 自定义 colormap
            n_colors = 256;
            c1 = [0.7, 0.9, 1.0]; % 浅天蓝
            c2 = [1.0, 0.0, 0.0]; % 纯红
            custom_map = [linspace(c1(1),c2(1),n_colors)', linspace(c1(2),c2(2),n_colors)', linspace(c1(3),c2(3),n_colors)'];
        
            for i = 1:4
                ax = nexttile;
                img_data = double(data{i});
                img_data(isnan(img_data)) = 0; 
                
                imagesc(lonV, latV, flipud(img_data));
                
                colormap(ax, custom_map); caxis(ax, [0 1]); 
                axis xy; axis image; grid on;
                set(ax, 'GridColor', [0.4 0.6 0.8], 'GridAlpha', 0.5);
                
                cb = colorbar('Location', 'southoutside');
                set(cb, 'Ticks', 0:0.2:1, 'FontSize', 9);
                
                xticks(linspace(min(lonV), max(lonV), 5));
                yticks(linspace(min(latV), max(latV), 5));
                xtickformat('%.3f°'); ytickformat('%.3f°');
                
                title(titles{i}, 'FontSize', 13, 'FontWeight', 'bold');
                xlabel('经度'); if i == 1, ylabel('纬度'); end
            end
            
            outFile = fullfile(outDir, '02_掩码集成_浅蓝红.png');
            fprintf('  Saving: %s\n', outFile);
            print(f, outFile, '-dpng', '-r400');
            close(f);
        end

        %% ================= 3. 深部预测图 (viz_deep_prediction) =================
        function run_deep_prediction(Au, lonG, latG, lonR, latR, lonT, latT, rIdx, mineral, outDir)
            f = figure('Position', [500 100 1200 1000], 'Color', 'w', 'Visible', 'off');
            
            lonV = linspace(min(lonG(:)), max(lonG(:)), size(Au, 2));
            latV = linspace(min(latG(:)), max(latG(:)), size(Au, 1));
            
            % 1. 潜力底图
            [~, hFill] = contourf(lonV, latV, flipud(double(Au)), 80, 'LineColor', 'none');
            hold on; colormap(jet(256));
            cb = colorbar; ylabel(cb, '潜力指数'); caxis([0.4 1]);
            
            % 2. 等值线
            [C, l] = contour(lonV, latV, flipud(double(Au)), 0.4:0.05:1, 'LineColor', [0.8 0.8 0.8]);
            clabel(C, l, 'FontSize', 8, 'Color', [0.9 0.9 0.9]);
        
            % 3. ROI 边界
            plot(lonR, latR, 'k-', 'LineWidth', 2.5); 
            
            % 4. 高潜力点
            if ~isempty(lonT)
                 plot(lonT, latT, 'wo', 'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.2 0.2]); 
            end
            
            % 5. 核心标定
            if ~isempty(rIdx) && ~isempty(lonT)
                plot(lonT(rIdx), latT(rIdx), 'yo', 'MarkerSize', 18, 'LineWidth', 3);
            end
            
            axis xy; axis image; grid on;
            title(['Deep Mineralization Prediction: ', upper(mineral)], 'FontSize', 16);
            xlabel('Longitude (°E)'); ylabel('Latitude (°N)');
            
            outFile = fullfile(outDir, '03_深部成矿预测图.png');
            fprintf('  Saving: %s\n', outFile);
            print(f, outFile, '-dpng', '-r500');
            close(f);
        end
    end
end