classdef GeoUtils
    methods (Static)
        %% ================= 1. 区域配置与路径获取 =================
        function [data_dir, roi_file, belt_lon, belt_lat] = getRegionConfig(region_type)
            % 根据 region_type 返回数据路径和 ROI 文件名
            % 注意：请根据你的实际路径修改这里
            switch region_type
                case 'shanxi'
                    % 示例路径，请修改为你电脑上的实际路径
                    % 假设你的项目结构是 Project/data/...
                    % 这里为了演示，指向一个假设路径，请确保改为 Main.m 中能找到的路径
                    % 如果你在 Main.m 中使用的是相对路径，这里需要保持一致
                    
                    % ⚠️ 请将此处修改为你真实的 data 文件夹路径 ⚠️
                    data_dir = 'C:\Users\Deep-Lei\Desktop\下载任务结果\新疆高昌区库格孜觉北金矿（四川黄金）59.05km2（20260104任务，20260105下载）\data';
                    roi_file = '经纬度坐标.xlsx';
                    
                % ... 你可以在这里添加其他 case ...
                otherwise
                    error('GeoUtils:UnknownRegion', '未知的区域类型: %s', region_type);
            end
            
            % 获取 root_dir (data 的上一级)
            root_dir = fileparts(data_dir);
            
            % 调用本类中的静态方法获取 belt 坐标
            [belt_lon, belt_lat] = GeoUtils.get_belt_coords(root_dir, roi_file);
        end

        function [belt_lon, belt_lat] = get_belt_coords(root_dir, roi_file)
            % 读取 ROI Excel 文件并生成边界矩形
            fullpath_roi = fullfile(root_dir, roi_file);
            if ~exist(fullpath_roi, 'file')
                error('GeoUtils:FileNotFound', '未找到ROI文件: %s', fullpath_roi);
            end
            
            opts = detectImportOptions(fullpath_roi);
            opts.VariableNamingRule = 'preserve';
            T_roi = readtable(fullpath_roi, opts);
            
            % 假设第2列是经度，第3列是纬度 (根据原代码逻辑)
            raw_data = table2cell(T_roi);
            lon_excel = cell2mat(raw_data(:, 2)); 
            lat_excel = cell2mat(raw_data(:, 3));
            
            valid_roi = ~isnan(lon_excel) & ~isnan(lat_excel) & isfinite(lon_excel) & isfinite(lat_excel);
            lon_valid = lon_excel(valid_roi);
            lat_valid = lat_excel(valid_roi);
            
            if isempty(lon_valid)
                error('ROI文件中没有有效坐标');
            end
            
            x1 = round(min(lon_valid), 2);
            x2 = round(max(lon_valid), 2);
            y1 = round(min(lat_valid), 2);
            y2 = round(max(lat_valid), 2);
            
            belt_lon = [x1 x2 x2 x1 x1];
            belt_lat = [y1 y1 y2 y2 y1];
        end

        %% ================= 2. 数据读取封装 =================
        function [s2, R] = readSentinel2(data_dir)
            s2_dir = fullfile(data_dir, 'Sentinel*2 L2*');
            files = dir(fullfile(s2_dir, '*B08*'));
            if isempty(files), files = dir(fullfile(s2_dir, '*.tif*')); end
            if isempty(files), files = dir(fullfile(s2_dir, '*.jp2')); end
            if isempty(files), error('GeoUtils:NoData', '未找到Sentinel-2 L2A文件'); end
            
            firstFile = fullfile(files(1).folder, files(1).name);
            [~, R] = readgeoraster(firstFile);
            
            % 定义波段模式
            s2_patterns = {{'B02'},{'B03'},{'B04'},{'B08'},{'B11'},{'B12'},{'B05'},{'B06'},{'B07'}};
            s2_raw = GeoUtils.readMultiBands_smart(files(1).folder, s2_patterns, R, 9);
            
            % 转换为反射率
            s2 = single(s2_raw) * 0.0001;
        end
        
        function lan = readLandsat8(data_dir, R)
            lan_l1 = dir(fullfile(data_dir, '*Landsat*8*L1*'));
            lan_l2 = dir(fullfile(data_dir, '*Landsat*8*L2*'));
            if ~isempty(lan_l1)
                lan_dir = fullfile(data_dir, lan_l1(1).name);
            elseif ~isempty(lan_l2)
                lan_dir = fullfile(data_dir, lan_l2(1).name);
            else
                error('GeoUtils:NoData', '未找到Landsat 8数据');
            end
            
            lan_patterns = {{'B2'},{'B3'},{'B4'},{'B5'},{'B6'},{'B7'},{'B8'}};
            lan = GeoUtils.readMultiBands_smart(lan_dir, lan_patterns, R, 7);
        end
        
        function ast = readASTER(data_dir, R)
            aster_dirs = dir(fullfile(data_dir, '*ASTER*'));
            if isempty(aster_dirs)
                error('GeoUtils:NoData', '未找到ASTER数据');
            end
            aster_dir = fullfile(data_dir, aster_dirs(1).name);
            
            % ASTER 波段映射
            aster_pat = {
                {'B01','B1'}, {'B02','B2'}, {'B3N','B03N'}, ... % 1-3
                {'B04','B4'}, {'B05','B5'}, {'B06','B6'}, {'B07','B7'}, {'B08','B8'}, {'B09'}, ... % 4-9
                {'B10'}, {'B11'}, {'B12'}, {'B13'}, {'B14'} % 10-14
            };
            
            [H, W] = deal(R.RasterSize(1), R.RasterSize(2));
            ast = nan(H, W, 14, 'single');
            
            for b = 1:14
                single_band = GeoUtils.readAny_smart(aster_dir, aster_pat{b}, R);
                if b <= 9
                    single_band = single_band * 0.01; % VNIR/SWIR
                    single_band(isinf(single_band)) = NaN;
                else
                    single_band = single_band * 0.1 + 300; % TIR
                    single_band(isinf(single_band)) = 300;
                end
                ast(:,:,b) = single_band;
            end
        end
        
        function [dem, inROI, lonGrid, latGrid, lonROI, latROI] = readDEMandROI(data_dir, roi_file, R)
            % 1. 读取 DEM
            dem_files = dir(fullfile(data_dir, 'DEM.tif'));
            if isempty(dem_files), dem_files = dir(fullfile(data_dir, 'DEM.tiff')); end
            
            [H, W] = deal(R.RasterSize(1), R.RasterSize(2));
            
            if ~isempty(dem_files)
                [dem_raw, ~] = readgeoraster(fullfile(dem_files(1).folder, dem_files(1).name));
                dem = single(dem_raw);
                dem(isinf(dem)) = NaN;
                if ndims(dem)>2, dem = dem(:,:,1); end
                
                % 如果尺寸不匹配则插值
                if ~isequal(size(dem), [H, W])
                    % 简化的插值逻辑，这里假设 R 覆盖范围一致
                    dem = imresize(dem, [H, W], 'bilinear');
                end
            else
                dem = nan(H, W, 'single');
                warning('GeoUtils:NoDEM', '未找到DEM文件，使用NaN填充');
            end
            
            % 2. 生成经纬度网格
            lonVec = linspace(R.LongitudeLimits(1), R.LongitudeLimits(2), W);
            latVec = linspace(R.LatitudeLimits(1), R.LatitudeLimits(2), H);
            [lonGrid, latGrid] = meshgrid(lonVec, latVec);
            
            % 3. 读取 ROI 并生成 Mask
            root_dir = fileparts(data_dir);
            fullpath = fullfile(root_dir, roi_file);
            opts = detectImportOptions(fullpath);
            opts.VariableNamingRule = 'preserve';
            T = readtable(fullpath, opts);
            raw = table2cell(T);
            lonROI = cell2mat(raw(:,2));
            latROI = cell2mat(raw(:,3));
            
            valid = ~isnan(lonROI) & ~isnan(latROI);
            lonROI = lonROI(valid); latROI = latROI(valid);
            
            if lonROI(1) ~= lonROI(end)
                lonROI(end+1) = lonROI(1);
                latROI(end+1) = latROI(1);
            end
            
            roiPoly = polyshape(lonROI, latROI);
            inROI_vec = isinterior(roiPoly, lonGrid(:), latGrid(:));
            inROI_1 = reshape(inROI_vec, H, W);
            
            % ⚠️ 关键：上下翻转 inROI 以匹配图像坐标系
            inROI = flipud(inROI_1);
        end

        %% ================= 3. 图像处理辅助函数 =================
        function cube = readMultiBands_smart(dirPath, patterns_cell, R, numBands)
            [H, W] = deal(R.RasterSize(1), R.RasterSize(2));
            cube = nan(H, W, numBands, 'single');
            files = [dir(fullfile(dirPath,'*.tif')); dir(fullfile(dirPath,'*.tiff')); dir(fullfile(dirPath,'*.jp2'))];
            
            for b = 1:numBands
                pattern = patterns_cell{b};
                for k = 1:length(files)
                    fname = upper(files(k).name);
                    % 检查是否匹配任一模式
                    if any(cellfun(@(p) contains(fname, upper(p)), pattern))
                        [A, ~] = readgeoraster(fullfile(files(k).folder, files(k).name));
                        if ndims(A)>2, A = A(:,:,1); end
                        A = single(A);
                        % 简单尺寸匹配
                        if ~isequal(size(A), [H, W])
                           A = imresize(A, [H, W], 'bilinear');
                        end
                        cube(:,:,b) = A;
                        break; 
                    end
                end
            end
        end
        
        function band = readAny_smart(dirPath, patterns_cell, R)
            % 复用 readMultiBands_smart 读取单波段
            cube = GeoUtils.readMultiBands_smart(dirPath, {patterns_cell}, R, 1);
            band = cube(:,:,1);
        end

        function band = getBand(varargin)
            % 从多个数据源中尝试提取波段
            idx = varargin{end};
            for i = 1:(length(varargin)-1)
                cube = varargin{i};
                if size(cube,3) >= idx && nnz(cube(:,:,idx)~=0 & ~isnan(cube(:,:,idx))) > 100
                    band = cube(:,:,idx); 
                    return;
                end
            end
            % 如果都没找到，返回NaN矩阵
            [H, W] = size(varargin{1}, [1, 2]);
            band = nan(H, W, 'single');
        end
        
        function img_norm = mat2gray_roi(img, inROI, min_val, max_val)
            img_norm = nan(size(img));
            img_roi = img(inROI);
            img_roi = img_roi(~isnan(img_roi) & ~isinf(img_roi));
            
            if isempty(img_roi), return; end
            
            if nargin < 3, min_val = min(img_roi); end
            if nargin < 4, max_val = max(img_roi); end
            
            if max_val - min_val < eps
                img_norm(inROI) = 0.5;
            else
                img_norm(inROI) = (img(inROI) - min_val) / (max_val - min_val);
                img_norm(img_norm < 0) = 0;
                img_norm(img_norm > 1) = 1;
            end
        end

        %% ================= 4. 算法核心函数 =================
        function [S2REP, REP_QA] = calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets)
            % 原逻辑实现
            B4_val = double(B4)/0.0001 * scale_factors(1) + offsets(1); % 还原并应用校正
            B5_val = double(B5)/0.0001 * scale_factors(2) + offsets(2);
            B6_val = double(B6)/0.0001 * scale_factors(3) + offsets(3);
            B7_val = double(B7)/0.0001 * scale_factors(4) + offsets(4);
            
            % 简化逻辑：直接使用反射率计算（假设输入已经是反射率，上面的DN还原可视情况调整）
            % 为保持一致性，如果输入已经是反射率，这里直接用：
            numerator = ((B4 + B7) / 2) - B5;
            denominator = (B6 - B5) + 1e-8;
            S2REP = 705 + 35 * (numerator ./ denominator);
            
            % 简单QA
            REP_QA = ones(size(S2REP));
            REP_QA(S2REP < 680 | S2REP > 760) = 4;
        end

        function F_abs = computeIntrinsicAbsorption(ast, mineral_type)
            % 基于矿物类型的本征吸收计算
            eps_val = 1e-6;
            [H, W, ~] = size(ast);
            F_abs = nan(H, W, 'single');
            
            % 提取常用波段索引以便阅读
            % ast 是 14个波段
            b1=ast(:,:,1); b2=ast(:,:,2); b3=ast(:,:,3); b4=ast(:,:,4);
            b5=ast(:,:,5); b6=ast(:,:,6); b7=ast(:,:,7); b8=ast(:,:,8);
            
            switch lower(mineral_type)
                case 'gold'
                    cont = (b3 + b5)/2; target = b3;
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.5 * (b6 ./ (b5 + eps_val));
                case 'copper'
                    cont = (b3 + b5)/2; target = b3;
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.5 * (b6 ./ (b5 + eps_val));
                case 'iron'
                    cont = (b2 + b4)/2; target = b3;
                    F_abs = (cont - target) ./ (cont + eps_val);
                otherwise
                    % 默认使用 OH 吸收
                    cont = (b5 + b7)/2; target = b6;
                    F_abs = (cont - target) ./ (cont + eps_val);
            end
            F_abs(isinf(F_abs)) = NaN;
        end
        
        function moran = computeMoranLocal(Z)
            % 计算 Moran I (无 ROI 限制)
            F_mean = mean(Z(:), 'omitnan');
            F_std = std(Z(:), 'omitnan');
            Z_norm = (Z - F_mean) ./ (F_std + eps);
            
            local_sum = GeoUtils.calc_local_sum_with_nan(Z_norm);
            moran = mat2gray(Z_norm .* local_sum);
            moran(isnan(moran)) = 0;
        end

        function moran = computeMoranLocalROI(F_abs, inROI)
            % 计算 Moran I (仅 ROI)
            F_roi = F_abs(inROI);
            F_mean = mean(F_roi, 'omitnan');
            F_std = std(F_roi, 'omitnan');
            if F_std == 0, F_std = eps; end
            
            Z = (F_abs - F_mean) / F_std;
            Z(~inROI) = NaN;
            Z(isinf(Z)) = 0;
            
            local_sum = GeoUtils.calc_local_sum_with_nan(Z);
            
            ls_roi = local_sum(inROI);
            max_ls = max(ls_roi, [], 'omitnan');
            if isempty(max_ls) || max_ls==0, max_ls = eps; end
            
            moran = Z .* local_sum / max_ls;
            moran(~inROI) = NaN;
            moran(isnan(moran)) = 0;
        end
        
        function local_sum = calc_local_sum_with_nan(Z)
            [rows, cols] = size(Z);
            pad = 1;
            Z_padded = padarray(Z, [pad, pad], NaN, 'both');
            local_sum = nan(rows, cols);
            
            % 3x3 权重，中心为0
            w = ones(3,3); w(2,2)=0;
            
            % 简单循环实现 (也可向量化，但循环对内存更友好)
            for i = 1:rows
                for j = 1:cols
                    if ~isnan(Z(i,j))
                        neigh = Z_padded(i:i+2, j:j+2);
                        mask = ~isnan(neigh);
                        if any(mask(:))
                            w_mask = w .* mask;
                            w_sum = sum(w_mask(:));
                            if w_sum > 0
                                w_mask = w_mask / w_sum;
                                local_sum(i,j) = sum(sum(neigh .* w_mask), 'omitnan');
                            else
                                local_sum(i,j) = 0;
                            end
                        else
                            local_sum(i,j) = 0;
                        end
                    end
                end
            end
        end
        
        %% ================= 5. 参数获取 =================
        function [F_thr, delta_thr, Moran_thr, enh_func] = getMineralThresholds(mineral_type)
            % 简化的阈值表，根据需要补充完整
            switch lower(mineral_type)
                case 'gold'
                    F_thr = 0.018; delta_thr = -2; Moran_thr = 0.20;
                    enh_func = @(Ferric, Fe_anom, OH_anom, Clay, NDVI) ...
                        0.45*Ferric + 0.25*Fe_anom + 0.15*OH_anom + 0.10*Clay + 0.05*NDVI;
                case 'copper'
                    F_thr = 0.020; delta_thr = -3; Moran_thr = 0.25;
                    enh_func = @(Ferric, Fe_anom, OH_anom, Clay, NDVI) ...
                        0.40*Clay + 0.30*OH_anom + 0.20*Ferric + 0.10*Fe_anom;
                case 'iron'
                    F_thr = 0.030; delta_thr = -4; Moran_thr = 0.35;
                    enh_func = @(Ferric, Fe_anom, OH_anom, Clay, NDVI) ...
                        0.60*Ferric + 0.40*Fe_anom;
                otherwise
                    % 默认金矿参数
                    F_thr = 0.018; delta_thr = -2; Moran_thr = 0.20;
                    enh_func = @(Ferric, Fe_anom, OH_anom, Clay, NDVI) ...
                        0.45*Ferric + 0.25*Fe_anom + 0.15*OH_anom + 0.10*Clay + 0.05*NDVI;
            end
        end
        
        function params = getYakymchukParams(mineral_type)
            switch lower(mineral_type)
                case 'gold',      params = struct('a',50, 'b',150, 'c',20);
                case 'copper',    params = struct('a',40, 'b',120, 'c',18);
                case 'iron',      params = struct('a',35, 'b',100, 'c',15);
                otherwise,        params = struct('a',50, 'b',150, 'c',20);
            end
        end
    end
end