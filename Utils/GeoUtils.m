classdef GeoUtils
    methods (Static)
        %% ================= 1. 区域配置与路径获取 =================
        function [data_dir, roi_file, belt_lon, belt_lat] = getRegionConfig(region_type)
            % 交互模式：强制手动选择
            fprintf('>>> [交互模式] 1. 请选择 data 数据文件夹...\n');
            sel_path = uigetdir(pwd, '请选择 data 数据文件夹');
            if sel_path == 0, error('用户取消了文件夹选择，程序终止。'); end
            data_dir = sel_path;
            
            % 默认从 data 的上级目录开始选，但用户可以随意跳转
            default_path = fileparts(data_dir);
            
            fprintf('>>> [交互模式] 2. 请选择经纬度坐标 Excel 文件...\n');
            [f_name, f_path] = uigetfile({'*.xlsx';'*.xls';'*.csv'}, '请选择经纬度坐标文件', default_path);
            if f_name == 0, error('用户取消了坐标文件选择，程序终止。'); end
            
            % [核心修改] 返回文件的绝对路径，确保后续读取不出错
            roi_file = fullfile(f_path, f_name);
            fprintf('    ✅ 已选择坐标文件: %s\n', roi_file);
            
            % 获取成矿带范围
            [belt_lon, belt_lat] = GeoUtils.get_belt_coords(f_path, f_name);
        end

        function [belt_lon, belt_lat] = get_belt_coords(root_dir, roi_file)
            % 组合全路径 (如果是文件名) 或者直接使用全路径
            if isfile(roi_file)
                fullpath_roi = roi_file;
            else
                fullpath_roi = fullfile(root_dir, roi_file);
            end
            
            % 使用智能读取获取有效坐标
            [~, ~, ~, ~, lon_valid, lat_valid] = GeoUtils.readROI_Robust(fullpath_roi);
            
            % 构造闭合的矩形框
            x1 = round(min(lon_valid), 4); x2 = round(max(lon_valid), 4);
            y1 = round(min(lat_valid), 4); y2 = round(max(lat_valid), 4);
            belt_lon = [x1 x2 x2 x1 x1]; 
            belt_lat = [y1 y1 y2 y2 y1];
            
            fprintf('    自动识别坐标范围: Lon[%.2f ~ %.2f], Lat[%.2f ~ %.2f]\n', x1, x2, y1, y2);
        end
        
        %% ================= 2. 智能 ROI 读取函数 (保留修复版) =================
        function [roiPoly, inROI_vec, lonGrid, latGrid, lonROI, latROI] = readROI_Robust(fullpath_roi)
             try
                 % 1. 读取所有原始内容
                 raw_data = readcell(fullpath_roi);
                 [rows, cols] = size(raw_data);
                 
                 % 2. 智能识别经纬度列
                 candidates_lon = []; candidates_lat = [];
                 parsed_cols = cell(1, cols);
                 
                 for c = 1:cols
                     col_data = raw_data(:, c);
                     nums = zeros(rows, 1);
                     valid_count = 0;
                     for r = 1:rows
                         val = col_data{r};
                         if isnumeric(val) && ~isnan(val)
                             nums(r) = val; valid_count = valid_count + 1;
                         elseif (ischar(val) || isstring(val)) 
                             d = str2double(val);
                             if ~isnan(d), nums(r) = d; valid_count = valid_count + 1; else, nums(r) = NaN; end
                         else, nums(r) = NaN; end
                     end
                     
                     if valid_count > 3
                         valid_nums = nums(~isnan(nums));
                         mean_val = mean(valid_nums);
                         min_val = min(valid_nums);
                         max_val = max(valid_nums);
                         parsed_cols{c} = nums;
                         
                         % 特征判断
                         if mean_val > 60 && mean_val < 160 && (max_val - min_val) < 20
                             candidates_lon = [candidates_lon, c];
                         end
                         if mean_val > 0 && mean_val < 60 && (max_val - min_val) < 20
                             candidates_lat = [candidates_lat, c];
                         end
                     end
                 end
                 
                 % 3. 决策逻辑
                 if length(candidates_lon) == 1 && length(candidates_lat) == 1
                     final_lon_col = candidates_lon; final_lat_col = candidates_lat;
                 else
                     % 默认第2列Lon, 第3列Lat (如果无法智能识别)
                     if cols >= 3 && ~isempty(parsed_cols{2}), final_lon_col = 2; final_lat_col = 3;
                     elseif cols >= 2 && ~isempty(parsed_cols{1}), final_lon_col = 1; final_lat_col = 2;
                     else, error('无法识别经纬度列，请检查文件格式。'); end
                 end
                 
                 lonROI = parsed_cols{final_lon_col};
                 latROI = parsed_cols{final_lat_col};
                 
                 valid_mask = ~isnan(lonROI) & ~isnan(latROI);
                 lonROI = lonROI(valid_mask);
                 latROI = latROI(valid_mask);
                 
                 if isempty(lonROI), error('没有有效坐标数据'); end
                 
                 % 闭合
                 if abs(lonROI(1) - lonROI(end)) > 1e-6 || abs(latROI(1) - latROI(end)) > 1e-6
                     lonROI(end+1) = lonROI(1); latROI(end+1) = latROI(1); 
                 end
                 
                 roiPoly = polyshape(lonROI, latROI);
                 inROI_vec = []; lonGrid=[]; latGrid=[];
                 
             catch ME
                 fprintf('读取坐标文件出错: %s\n', ME.message);
                 rethrow(ME);
             end
        end

        %% ================= 3. 数据读取封装 =================
        function [s2, R, ref_tif_path] = readSentinel2(data_dir)
            s2_dir = fullfile(data_dir, 'Sentinel*2 L2*');
            files = dir(fullfile(s2_dir, '*B08*'));
            if isempty(files), files = dir(fullfile(s2_dir, '*.tif*')); end
            if isempty(files), files = dir(fullfile(s2_dir, '*.jp2')); end
            if isempty(files), error('GeoUtils:NoData', '未找到Sentinel-2 L2A文件'); end
            
            firstFile = fullfile(files(1).folder, files(1).name);
            ref_tif_path = firstFile;
            [~, R] = readgeoraster(firstFile);
            
            s2_patterns = {{'B02'},{'B03'},{'B04'},{'B08'},{'B11'},{'B12'},{'B05'},{'B06'},{'B07'}};
            s2_raw = GeoUtils.readMultiBands_smart(files(1).folder, s2_patterns, R, 9);
            s2 = single(s2_raw) * 0.0001;
        end
        
        function lan = readLandsat8(data_dir, R)
            lan_l1 = dir(fullfile(data_dir, '*Landsat*8*L1*'));
            lan_l2 = dir(fullfile(data_dir, '*Landsat*8*L2*'));
            if ~isempty(lan_l1), lan_dir = fullfile(data_dir, lan_l1(1).name);
            elseif ~isempty(lan_l2), lan_dir = fullfile(data_dir, lan_l2(1).name);
            else, error('GeoUtils:NoData', '未找到Landsat 8数据'); end
            lan_patterns = {{'B2'},{'B3'},{'B4'},{'B5'},{'B6'},{'B7'},{'B8'}};
            lan = GeoUtils.readMultiBands_smart(lan_dir, lan_patterns, R, 7);
        end
        
        function ast = readASTER(data_dir, R)
            aster_dirs = [dir(fullfile(data_dir, '*ASTER*L2*')); dir(fullfile(data_dir, '*ASTER*L1*'))];
            if isempty(aster_dirs), error('GeoUtils:NoData', '未找到ASTER数据'); end
            aster_dir = fullfile(data_dir, aster_dirs(1).name);
            aster_pat = {{'B01','B1'},{'B02','B2'},{'B3N','B03N'},{'B04','B4'},{'B05','B5'},{'B06','B6'},{'B07','B7'},{'B08','B8'},{'B09'},{'B10'},{'B11'},{'B12'},{'B13'},{'B14'}};
            [H, W] = deal(R.RasterSize(1), R.RasterSize(2));
            ast = nan(H, W, 14, 'single');
            for b = 1:14
                single_band = GeoUtils.readAny_smart(aster_dir, aster_pat{b}, R);
                if b <= 9, single_band = single_band * 0.01; single_band(isinf(single_band)) = NaN;
                else, single_band = single_band * 0.1 + 300; single_band(isinf(single_band)) = 300; end
                ast(:,:,b) = single_band;
            end
        end

        function [dem, inROI, lonGrid, latGrid, lonROI, latROI] = readDEMandROI(data_dir, roi_file, R)
            dem_files = dir(fullfile(data_dir, 'DEM.tif'));
            if isempty(dem_files), dem_files = dir(fullfile(data_dir, 'DEM.tiff')); end
            [H, W] = deal(R.RasterSize(1), R.RasterSize(2));
            lonVec = linspace(R.LongitudeLimits(1), R.LongitudeLimits(2), W);
            latVec = linspace(R.LatitudeLimits(1), R.LatitudeLimits(2), H);
            [lonGrid, latGrid] = meshgrid(lonVec, latVec);
            
            % [核心修改] 兼容绝对路径
            if isfile(roi_file)
                fullpath = roi_file;
            else
                root_dir = fileparts(data_dir);
                fullpath = fullfile(root_dir, roi_file);
            end
            
            % 调用智能读取
            [roiPoly, ~, ~, ~, lonROI, latROI] = GeoUtils.readROI_Robust(fullpath);
            
            inROI_vec = isinterior(roiPoly, lonGrid(:), latGrid(:));
            inROI = flipud(reshape(inROI_vec, H, W)); 
            
            if ~isempty(dem_files)
                [dem_raw, ~] = readgeoraster(fullfile(dem_files(1).folder, dem_files(1).name));
                dem = single(dem_raw); dem(isinf(dem)) = NaN;
                if ndims(dem)>2, dem = dem(:,:,1); end
                if ~isequal(size(dem), [H, W])
                    lon0 = linspace(R.LongitudeLimits(1),R.LongitudeLimits(2),size(dem,2));
                    lat0 = linspace(R.LatitudeLimits(1),R.LatitudeLimits(2),size(dem,1));
                    [Xq,Yq] = meshgrid(linspace(R.LongitudeLimits(1),R.LongitudeLimits(2),W), linspace(R.LatitudeLimits(1),R.LatitudeLimits(2),H));
                    dem = interp2(lon0, lat0, dem, Xq, Yq, 'linear', NaN);
                end
            else, dem = nan(H, W, 'single'); end
        end

        % 辅助函数保持不变
        function cube = readMultiBands_smart(dirPath, patterns_cell, R, numBands)
            [H, W] = deal(R.RasterSize(1), R.RasterSize(2));
            cube = nan(H, W, numBands, 'single');
            files = [dir(fullfile(dirPath,'*.tif')); dir(fullfile(dirPath,'*.tiff')); dir(fullfile(dirPath,'*.jp2'))];
            for b = 1:numBands
                pattern = patterns_cell{b};
                for k = 1:length(files)
                    fname = upper(files(k).name);
                    if any(cellfun(@(p) contains(fname, upper(p)), pattern))
                        [A, ~] = readgeoraster(fullfile(files(k).folder, files(k).name));
                        if ndims(A)>2, A = A(:,:,1); end
                        A = single(A);
                        if ~isequal(size(A), [H, W])
                            lon0 = linspace(R.LongitudeLimits(1),R.LongitudeLimits(2),size(A,2));
                            lat0 = linspace(R.LatitudeLimits(1),R.LatitudeLimits(2),size(A,1));
                            [Xq,Yq] = meshgrid(linspace(R.LongitudeLimits(1),R.LongitudeLimits(2),W), linspace(R.LatitudeLimits(1),R.LatitudeLimits(2),H));
                            A = interp2(lon0, lat0, A, Xq, Yq, 'linear', NaN);
                        end
                        cube(:,:,b) = A; break; 
                    end
                end
            end
        end
        function band = readAny_smart(dirPath, patterns_cell, R)
            if ischar(patterns_cell), patterns_cell = {patterns_cell}; end
            cube = GeoUtils.readMultiBands_smart(dirPath, {patterns_cell}, R, 1);
            band = cube(:,:,1);
        end
        function band = getBand(varargin)
            idx = varargin{end};
            for i = 1:(length(varargin)-1)
                cube = varargin{i};
                if size(cube,3) >= idx && nnz(cube(:,:,idx)~=0 & ~isnan(cube(:,:,idx))) > 100, band = cube(:,:,idx); return; end
            end
            [H, W] = size(varargin{1}, [1, 2]); band = nan(H, W, 'single');
        end
        function img_norm = mat2gray_roi(img, inROI, min_val, max_val)
            img_norm = nan(size(img)); img_roi = img(inROI); img_roi = img_roi(~isnan(img_roi) & ~isinf(img_roi));
            if isempty(img_roi), return; end
            if nargin < 3, min_val = min(img_roi); end
            if nargin < 4, max_val = max(img_roi); end
            if max_val - min_val < eps, img_norm(inROI) = 0.5; else, val = (img(inROI) - min_val) / (max_val - min_val); val(val < 0) = 0; val(val > 1) = 1; img_norm(inROI) = val; end
        end
        function [S2REP, REP_QA] = calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets)
            B4_val = double(B4 * 10000) * scale_factors(1) + offsets(1); B5_val = double(B5 * 10000) * scale_factors(2) + offsets(2);
            B6_val = double(B6 * 10000) * scale_factors(3) + offsets(3); B7_val = double(B7 * 10000) * scale_factors(4) + offsets(4);
            invalid_reflect = (B4_val < 0 | B4_val > 1) | (B5_val < 0 | B5_val > 1) | (B6_val < 0 | B6_val > 1) | (B7_val < 0 | B7_val > 1) | isnan(B4_val) | isnan(B5_val) | isnan(B6_val) | isnan(B7_val);
            [H, W] = size(B4); S2REP = nan(H, W); REP_QA = zeros(H, W); REP_QA(invalid_reflect) = 3; valid_pixel = ~invalid_reflect;
            numerator = ((B4_val + B7_val) / 2) - B5_val; denominator = (B6_val - B5_val) + 1e-8;
            zero_denominator = valid_pixel & (abs(denominator) < 1e-6); REP_QA(zero_denominator) = 2; valid_pixel(zero_denominator) = false;
            S2REP(valid_pixel) = 705 + 35 * (numerator(valid_pixel) ./ denominator(valid_pixel));
            rep_out_range = valid_pixel & (S2REP < 680 | S2REP > 760); REP_QA(rep_out_range) = 4; S2REP(rep_out_range) = NaN; REP_QA(valid_pixel & ~rep_out_range) = 1;
        end
        function F_abs = computeIntrinsicAbsorption(ast, mineral_type)
            eps_val = 1e-6; [H, W, ~] = size(ast); F_abs = nan(H, W, 'single');
            switch lower(mineral_type)
                case 'gold', cont = (ast(:,:,3) + ast(:,:,5))/2; target = ast(:,:,3); F_abs = (cont - target) ./ (cont + eps_val); F_abs = F_abs + 0.5 * (ast(:,:,6) ./ (ast(:,:,5) + eps_val));
                otherwise, cont = (ast(:,:,5) + ast(:,:,7))/2; target = ast(:,:,6); F_abs = (cont - target) ./ (cont + eps_val);
            end
            F_abs(isinf(F_abs)) = NaN;
        end
        function indices = computeDEMIndices(dem, mineral_type, H, W, inROI)
             indices = struct('slope', nan(H,W,'single'), 'neg_curvature', nan(H,W,'single'));
             if ~strcmpi(mineral_type, 'cave'), return; end
             [dx, dy] = gradient(dem); slope = atan(sqrt(dx.^2 + dy.^2)) * 180 / pi; indices.slope = GeoUtils.mat2gray_roi(slope, inROI); 
             [dxx, ~] = gradient(dx); [~, dyy] = gradient(dy); curvature = -(dxx + dyy); neg_curv = max(-curvature, 0); indices.neg_curvature = GeoUtils.mat2gray_roi(neg_curv, inROI); 
        end
        function local_sum = calc_local_sum_with_nan(Z)
            [rows, cols] = size(Z); pad = 1; Z_padded = padarray(Z, [pad, pad], NaN, 'both'); local_sum = nan(rows, cols); w = ones(3,3); w(2,2)=0;
            for i = 1:rows, for j = 1:cols, if ~isnan(Z(i,j)), neigh = Z_padded(i:i+2, j:j+2); mask = ~isnan(neigh);
            if any(mask(:)), w_mask = w .* mask; w_sum = sum(w_mask(:)); if w_sum > 0, w_mask = w_mask / w_sum; local_sum(i,j) = sum(sum(neigh .* w_mask), 'omitnan'); else, local_sum(i,j) = 0; end; else, local_sum(i,j) = 0; end; end; end; end
            local_sum(isinf(local_sum)) = NaN;
        end
        function [F_thr, delta_thr, Moran_thr, enh_func] = getMineralThresholds(mineral_type)
             F_thr = 0.018; delta_thr = -2; Moran_thr = 0.20;
             enh_func = @(Ferric, Fe_anom, OH_anom, Clay, NDVI) 0.45*Ferric + 0.25*Fe_anom + 0.15*OH_anom + 0.10*Clay + 0.05*NDVI;
        end
        function params = getYakymchukParams(mineral_type)
            params = struct('a',50, 'b',150, 'c',20);
        end
    end
end