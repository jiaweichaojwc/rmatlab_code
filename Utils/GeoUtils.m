classdef GeoUtils
    methods (Static)
        %% ================= 1. 区域配置与路径获取 =================
        function [data_dir, roi_file, belt_lon, belt_lat] = getRegionConfig(region_type)
            % 交互式逻辑：如果 region_type 为空，弹出选择框
            if isempty(region_type)
                fprintf('>>> [交互模式] 请选择 data 数据文件夹...\n');
                sel_path = uigetdir(pwd, '请选择 data 数据文件夹');
                if sel_path == 0, error('用户取消了文件夹选择，程序终止。'); end
                data_dir = sel_path;
                
                % 自动寻找 ROI 文件
                root_dir = fileparts(data_dir);
                possible_rois = dir(fullfile(root_dir, '*坐标*.xlsx'));
                if ~isempty(possible_rois)
                    roi_file = possible_rois(1).name;
                    fprintf('>>> 自动找到坐标文件: %s\n', roi_file);
                    f_path = root_dir;
                else
                    fprintf('>>> 未自动找到坐标文件，请手动选择...\n');
                    [f_name, f_path] = uigetfile({'*.xlsx';'*.xls'}, '请选择经纬度坐标文件', root_dir);
                    if f_name == 0, error('用户取消了坐标文件选择，程序终止。'); end
                    roi_file = f_name;
                end
                
                [belt_lon, belt_lat] = GeoUtils.get_belt_coords(f_path, roi_file);
                return;
            end

            % 硬编码区域逻辑 (保留以防万一)
             error('GeoUtils:UnknownRegion', '未知的区域类型: %s', region_type);
        end

        function [belt_lon, belt_lat] = get_belt_coords(root_dir, roi_file)
            fullpath_roi = fullfile(root_dir, roi_file);
            opts = detectImportOptions(fullpath_roi); opts.VariableNamingRule = 'preserve';
            T_roi = readtable(fullpath_roi, opts);
            raw_data = table2cell(T_roi);
            lon_excel = cell2mat(raw_data(:, 2)); lat_excel = cell2mat(raw_data(:, 3));
            valid_roi = ~isnan(lon_excel) & ~isnan(lat_excel) & isfinite(lon_excel) & isfinite(lat_excel);
            lon_valid = lon_excel(valid_roi); lat_valid = lat_excel(valid_roi);
            x1 = round(min(lon_valid), 2); x2 = round(max(lon_valid), 2);
            y1 = round(min(lat_valid), 2); y2 = round(max(lat_valid), 2);
            belt_lon = [x1 x2 x2 x1 x1]; belt_lat = [y1 y1 y2 y2 y1];
        end

        %% ================= 2. 数据读取封装 =================
        function [s2, R, ref_tif_path] = readSentinel2(data_dir)
            s2_dir = fullfile(data_dir, 'Sentinel*2 L2*');
            files = dir(fullfile(s2_dir, '*B08*'));
            if isempty(files), files = dir(fullfile(s2_dir, '*.tif*')); end
            if isempty(files), files = dir(fullfile(s2_dir, '*.jp2')); end
            if isempty(files), error('GeoUtils:NoData', '未找到Sentinel-2 L2A文件'); end
            
            % [关键修改] 获取第一张影像的完整路径，用于 KML 生成器的参考
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
            
            aster_pat = {
                {'B01','B1'}, {'B02','B2'}, {'B3N','B03N'}, ... 
                {'B04','B4'}, {'B05','B5'}, {'B06','B6'}, {'B07','B7'}, {'B08','B8'}, {'B09'}, ...
                {'B10'}, {'B11'}, {'B12'}, {'B13'}, {'B14'}
            };
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
            
            root_dir = fileparts(data_dir);
            fullpath = fullfile(root_dir, roi_file);
            opts = detectImportOptions(fullpath); opts.VariableNamingRule = 'preserve';
            T = readtable(fullpath, opts);
            raw = table2cell(T);
            lonROI = cell2mat(raw(:,2)); latROI = cell2mat(raw(:,3));
            valid = ~isnan(lonROI) & ~isnan(latROI);
            lonROI = lonROI(valid); latROI = latROI(valid);
            if lonROI(1) ~= lonROI(end), lonROI(end+1) = lonROI(1); latROI(end+1) = latROI(1); end
            
            roiPoly = polyshape(lonROI, latROI);
            inROI_vec = isinterior(roiPoly, lonGrid(:), latGrid(:));
            inROI = flipud(reshape(inROI_vec, H, W)); 

            if ~isempty(dem_files)
                [dem_raw, ~] = readgeoraster(fullfile(dem_files(1).folder, dem_files(1).name));
                dem = single(dem_raw); dem(isinf(dem)) = NaN;
                if ndims(dem)>2, dem = dem(:,:,1); end
                if ~isequal(size(dem), [H, W])
                    lon0 = linspace(R.LongitudeLimits(1),R.LongitudeLimits(2),size(dem,2));
                    lat0 = linspace(R.LatitudeLimits(1),R.LatitudeLimits(2),size(dem,1));
                    [Xq,Yq] = meshgrid(linspace(R.LongitudeLimits(1),R.LongitudeLimits(2),W), ...
                                       linspace(R.LatitudeLimits(1),R.LatitudeLimits(2),H));
                    dem = interp2(lon0, lat0, dem, Xq, Yq, 'linear', NaN);
                end
            else
                dem = nan(H, W, 'single');
            end
        end

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
                            [Xq,Yq] = meshgrid(linspace(R.LongitudeLimits(1),R.LongitudeLimits(2),W), ...
                                               linspace(R.LatitudeLimits(1),R.LatitudeLimits(2),H));
                            A = interp2(lon0, lat0, A, Xq, Yq, 'linear', NaN);
                        end
                        cube(:,:,b) = A;
                        break; 
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
                if size(cube,3) >= idx && nnz(cube(:,:,idx)~=0 & ~isnan(cube(:,:,idx))) > 100
                    band = cube(:,:,idx); return;
                end
            end
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
            if max_val - min_val < eps, img_norm(inROI) = 0.5;
            else, val = (img(inROI) - min_val) / (max_val - min_val); val(val < 0) = 0; val(val > 1) = 1; img_norm(inROI) = val; end
        end

        function [S2REP, REP_QA] = calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets)
            B4_val = double(B4 * 10000) * scale_factors(1) + offsets(1);
            B5_val = double(B5 * 10000) * scale_factors(2) + offsets(2);
            B6_val = double(B6 * 10000) * scale_factors(3) + offsets(3);
            B7_val = double(B7 * 10000) * scale_factors(4) + offsets(4);
            invalid_reflect = (B4_val < 0 | B4_val > 1) | (B5_val < 0 | B5_val > 1) ...
                            | (B6_val < 0 | B6_val > 1) | (B7_val < 0 | B7_val > 1) ...
                            | isnan(B4_val) | isnan(B5_val) | isnan(B6_val) | isnan(B7_val);
            [H, W] = size(B4); S2REP = nan(H, W); REP_QA = zeros(H, W); REP_QA(invalid_reflect) = 3; valid_pixel = ~invalid_reflect;
            numerator = ((B4_val + B7_val) / 2) - B5_val; denominator = (B6_val - B5_val) + 1e-8;
            zero_denominator = valid_pixel & (abs(denominator) < 1e-6); REP_QA(zero_denominator) = 2; valid_pixel(zero_denominator) = false;
            S2REP(valid_pixel) = 705 + 35 * (numerator(valid_pixel) ./ denominator(valid_pixel));
            rep_out_range = valid_pixel & (S2REP < 680 | S2REP > 760); REP_QA(rep_out_range) = 4; S2REP(rep_out_range) = NaN;
            REP_QA(valid_pixel & ~rep_out_range) = 1;
        end
        
        function F_abs = computeIntrinsicAbsorption(ast, mineral_type)
            eps_val = 1e-6; [H, W, ~] = size(ast); F_abs = nan(H, W, 'single');
            % 简化版，实际应包含所有case，与untitled3.m一致
            cont = (ast(:,:,5) + ast(:,:,7))/2; target = ast(:,:,6);
            F_abs = (cont - target) ./ (cont + eps_val);
            F_abs(isinf(F_abs)) = NaN;
        end

        function indices = computeDEMIndices(dem, mineral_type, H, W, inROI)
             indices = struct('slope', nan(H,W,'single'), 'neg_curvature', nan(H,W,'single'));
             if ~strcmpi(mineral_type, 'cave'), return; end
             [dx, dy] = gradient(dem); slope = atan(sqrt(dx.^2 + dy.^2)) * 180 / pi;
             indices.slope = GeoUtils.mat2gray_roi(slope, inROI); 
             [dxx, ~] = gradient(dx); [~, dyy] = gradient(dy); curvature = -(dxx + dyy);
             neg_curv = max(-curvature, 0); indices.neg_curvature = GeoUtils.mat2gray_roi(neg_curv, inROI); 
        end
        
        function local_sum = calc_local_sum_with_nan(Z)
            [rows, cols] = size(Z); pad = 1; Z_padded = padarray(Z, [pad, pad], NaN, 'both'); local_sum = nan(rows, cols); w = ones(3,3); w(2,2)=0;
            for i = 1:rows, for j = 1:cols, if ~isnan(Z(i,j)), neigh = Z_padded(i:i+2, j:j+2); mask = ~isnan(neigh);
            if any(mask(:)), w_mask = w .* mask; w_sum = sum(w_mask(:)); if w_sum > 0, w_mask = w_mask / w_sum; local_sum(i,j) = sum(sum(neigh .* w_mask), 'omitnan'); else, local_sum(i,j) = 0; end
            else, local_sum(i,j) = 0; end, end, end, end
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