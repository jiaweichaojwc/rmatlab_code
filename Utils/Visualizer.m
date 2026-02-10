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
            
            for i=1:8, ax=nexttile; imagesc(lonV, latV, flipud(data_list{i})); axis xy; axis image; grid on; title(titles{i}); 
            if i>1, colormap(ax,turbo(256)); colorbar; if ~isempty(clims{i}), caxis(ax, clims{i}); end; end, end
            print(f, fullfile(outDir, '01_共振参数综合图.png'), '-dpng', '-r300'); close(f);
        end

        % [关键修改] 动态排版支持 4 或 5 张掩码
        function run_mask_fusion(mask_list, title_list, lonGrid, latGrid, outDir)
            num_masks = length(mask_list);
            if num_masks == 0, return; end
            
            lonV = linspace(min(lonGrid(:)), max(lonGrid(:)), size(mask_list{1}, 2));
            latV = linspace(min(latGrid(:)), max(latGrid(:)), size(mask_list{1}, 1));
            
            % 智能排版：<=4张一行，>4张换行 (适应5张图)
            if num_masks <= 4, cols = num_masks; rows = 1;
            else, cols = 4; rows = ceil(num_masks / cols); end
            
            f = figure('Position', [100 450 400*cols 400*rows+50], 'Color', 'w', 'Visible', 'off');
            t = tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'tight');
            
            n_colors = 256;
            % 浅蓝 -> 红
            custom_map = [linspace(1,0,n_colors)', linspace(1,0.8,n_colors)', linspace(1,0,n_colors)'];

            for i = 1:num_masks
                ax = nexttile;
                img_data = double(mask_list{i});
                img_data(isnan(img_data)) = 0; 
                imagesc(lonV, latV, flipud(img_data));
                
                colormap(ax, custom_map); caxis(ax, [0 1]); axis xy; axis image; grid on;
                if i <= length(title_list), title(title_list{i}, 'FontSize', 12, 'FontWeight', 'bold'); end
            end
            
            outName = fullfile(outDir, sprintf('02_掩码集成_%d图.png', num_masks));
            print(f, outName, '-dpng', '-r400');
            close(f);
        end

        function run_deep_prediction(Au, lonG, latG, lonR, latR, lonT, latT, rIdx, mineral, outDir)
            f = figure('Position', [500 100 1200 1000], 'Color', 'w', 'Visible', 'off');
            lonV = linspace(min(lonG(:)), max(lonG(:)), size(Au, 2));
            latV = linspace(min(latG(:)), max(latG(:)), size(Au, 1));
            
            [~, hFill] = contourf(lonV, latV, flipud(double(Au)), 80, 'LineColor', 'none');
            hold on; colormap(jet(256)); colorbar; caxis([0.4 1]);
            contour(lonV, latV, flipud(double(Au)), 0.4:0.05:1, 'LineColor', [0.8 0.8 0.8]);
            plot(lonR, latR, 'k-', 'LineWidth', 2.5); 
            if ~isempty(lonT), plot(lonT, latT, 'wo', 'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.2 0.2]); end
            if ~isempty(rIdx), plot(lonT(rIdx), latT(rIdx), 'yo', 'MarkerSize', 18, 'LineWidth', 3); end
            
            axis xy; axis image; grid on; title(['Deep Prediction: ', upper(mineral)], 'FontSize', 16);
            print(f, fullfile(outDir, '03_深部成矿预测图.png'), '-dpng', '-r500'); close(f);
        end
    end
end