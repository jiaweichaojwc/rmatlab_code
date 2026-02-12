classdef GeoUtils
    methods (Static)
        %% ================= 1. 区域配置与路径获取 =================
        function [data_dir, roi_file, belt_lon, belt_lat] = getRegionConfig(region_type)
            % 交互模式：强制手动选择
            fprintf('>>> [交互模式] 1. 请选择 data 数据文件夹...\n');
            sel_path = uigetdir(pwd, '请选择 data 数据文件夹');
            
            if sel_path == 0
                error('用户取消了文件夹选择，程序终止。'); 
            end
            data_dir = sel_path;
            
            % 默认从 data 的上级目录开始选，但用户可以随意跳转
            default_path = fileparts(data_dir);
            
            fprintf('>>> [交互模式] 2. 请选择经纬度坐标 Excel 文件...\n');
            [f_name, f_path] = uigetfile({'*.xlsx';'*.xls';'*.csv'}, '请选择经纬度坐标文件', default_path);
            
            if f_name == 0
                error('用户取消了坐标文件选择，程序终止。'); 
            end
            
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
            x1 = round(min(lon_valid), 4); 
            x2 = round(max(lon_valid), 4);
            y1 = round(min(lat_valid), 4); 
            y2 = round(max(lat_valid), 4);
            
            belt_lon = [x1 x2 x2 x1 x1]; 
            belt_lat = [y1 y1 y2 y2 y1];
            
            fprintf('    自动识别坐标范围: Lon[%.2f ~ %.2f], Lat[%.2f ~ %.2f]\n', x1, x2, y1, y2);
        end
        
        %% ================= 2. 智能 ROI 读取函数 =================
        function [roiPoly, inROI_vec, lonGrid, latGrid, lonROI, latROI] = readROI_Robust(fullpath_roi)
             try
                 % 1. 读取所有原始内容
                 raw_data = readcell(fullpath_roi);
                 [rows, cols] = size(raw_data);
                 
                 % 2. 智能识别经纬度列
                 candidates_lon = []; 
                 candidates_lat = [];
                 parsed_cols = cell(1, cols);
                 
                 for c = 1:cols
                     col_data = raw_data(:, c);
                     nums = zeros(rows, 1);
                     valid_count = 0;
                     
                     for r = 1:rows
                         val = col_data{r};
                         if isnumeric(val) && ~isnan(val)
                             nums(r) = val; 
                             valid_count = valid_count + 1;
                         elseif (ischar(val) || isstring(val)) 
                             d = str2double(val);
                             if ~isnan(d)
                                 nums(r) = d; 
                                 valid_count = valid_count + 1; 
                             else
                                 nums(r) = NaN; 
                             end
                         else
                             nums(r) = NaN; 
                         end
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
                     final_lon_col = candidates_lon; 
                     final_lat_col = candidates_lat;
                 else
                     if cols >= 3 && ~isempty(parsed_cols{2})
                         final_lon_col = 2; final_lat_col = 3;
                     elseif cols >= 2 && ~isempty(parsed_cols{1})
                         final_lon_col = 1; final_lat_col = 2;
                     else
                         error('无法识别经纬度列，请检查文件格式。'); 
                     end
                 end
                 
                 lonROI = parsed_cols{final_lon_col};
                 latROI = parsed_cols{final_lat_col};
                 
                 valid_mask = ~isnan(lonROI) & ~isnan(latROI);
                 lonROI = lonROI(valid_mask);
                 latROI = latROI(valid_mask);
                 
                 if isempty(lonROI)
                     error('没有有效坐标数据'); 
                 end
                 
                 % 闭合
                 if abs(lonROI(1) - lonROI(end)) > 1e-6 || abs(latROI(1) - latROI(end)) > 1e-6
                     lonROI(end+1) = lonROI(1); 
                     latROI(end+1) = latROI(1); 
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
            
            if isempty(files)
                error('GeoUtils:NoData', '未找到Sentinel-2 L2A文件'); 
            end
            
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
            aster_dirs = [dir(fullfile(data_dir, '*ASTER*L2*')); dir(fullfile(data_dir, '*ASTER*L1*'))];
            if isempty(aster_dirs)
                error('GeoUtils:NoData', '未找到ASTER数据'); 
            end
            
            aster_dir = fullfile(data_dir, aster_dirs(1).name);
            aster_pat = {{'B01','B1'}, {'B02','B2'}, {'B3N','B03N'}, ...
                         {'B04','B4'}, {'B05','B5'}, {'B06','B6'}, ...
                         {'B07','B7'}, {'B08','B8'}, {'B09'}, {'B10'}, ...
                         {'B11'}, {'B12'}, {'B13'}, {'B14'}};
            
            [H, W] = deal(R.RasterSize(1), R.RasterSize(2));
            ast = nan(H, W, 14, 'single');
            
            for b = 1:14
                single_band = GeoUtils.readAny_smart(aster_dir, aster_pat{b}, R);
                
                if b <= 9
                    single_band = single_band * 0.01; 
                    single_band(isinf(single_band)) = NaN;
                else
                    single_band = single_band * 0.1 + 300; 
                    single_band(isinf(single_band)) = 300; 
                end
                ast(:,:,b) = single_band;
            end
        end

        function [dem, inROI, lonGrid, latGrid, lonROI, latROI] = readDEMandROI(data_dir, roi_file, R)
            dem_files = dir(fullfile(data_dir, 'DEM.tif'));
            if isempty(dem_files)
                dem_files = dir(fullfile(data_dir, 'DEM.tiff')); 
            end
            
            [H, W] = deal(R.RasterSize(1), R.RasterSize(2));
            lonVec = linspace(R.LongitudeLimits(1), R.LongitudeLimits(2), W);
            latVec = linspace(R.LatitudeLimits(1), R.LatitudeLimits(2), H);
            [lonGrid, latGrid] = meshgrid(lonVec, latVec);
            
            if isfile(roi_file)
                fullpath = roi_file;
            else
                root_dir = fileparts(data_dir);
                fullpath = fullfile(root_dir, roi_file);
            end
            
            [roiPoly, ~, ~, ~, lonROI, latROI] = GeoUtils.readROI_Robust(fullpath);
            
            inROI_vec = isinterior(roiPoly, lonGrid(:), latGrid(:));
            inROI = flipud(reshape(inROI_vec, H, W)); 
            
            if ~isempty(dem_files)
                [dem_raw, ~] = readgeoraster(fullfile(dem_files(1).folder, dem_files(1).name));
                dem = single(dem_raw); 
                dem(isinf(dem)) = NaN;
                
                if ndims(dem) > 2, dem = dem(:,:,1); end
                
                if ~isequal(size(dem), [H, W])
                    lon0 = linspace(R.LongitudeLimits(1), R.LongitudeLimits(2), size(dem,2));
                    lat0 = linspace(R.LatitudeLimits(1), R.LatitudeLimits(2), size(dem,1));
                    [Xq, Yq] = meshgrid(linspace(R.LongitudeLimits(1), R.LongitudeLimits(2), W), ...
                                        linspace(R.LatitudeLimits(1), R.LatitudeLimits(2), H));
                    dem = interp2(lon0, lat0, dem, Xq, Yq, 'linear', NaN);
                end
            else
                dem = nan(H, W, 'single'); 
            end
        end

        %% ================= 4. 辅助函数 =================
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
                        
                        if ndims(A) > 2, A = A(:,:,1); end
                        A = single(A);
                        
                        if ~isequal(size(A), [H, W])
                            lon0 = linspace(R.LongitudeLimits(1), R.LongitudeLimits(2), size(A,2));
                            lat0 = linspace(R.LatitudeLimits(1), R.LatitudeLimits(2), size(A,1));
                            [Xq, Yq] = meshgrid(linspace(R.LongitudeLimits(1), R.LongitudeLimits(2), W), ...
                                                linspace(R.LatitudeLimits(1), R.LatitudeLimits(2), H));
                            A = interp2(lon0, lat0, A, Xq, Yq, 'linear', NaN);
                        end
                        
                        cube(:,:,b) = A; 
                        break; 
                    end
                end
            end
        end
        
        function band = readAny_smart(dirPath, patterns_cell, R)
            if ischar(patterns_cell)
                patterns_cell = {patterns_cell}; 
            end
            cube = GeoUtils.readMultiBands_smart(dirPath, {patterns_cell}, R, 1);
            band = cube(:,:,1);
        end
        
        function band = getBand(varargin)
            idx = varargin{end};
            
            for i = 1:(length(varargin)-1)
                cube = varargin{i};
                if size(cube,3) >= idx && nnz(cube(:,:,idx)~=0 & ~isnan(cube(:,:,idx))) > 100
                    band = cube(:,:,idx); 
                    return; 
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
            
            if max_val - min_val < eps
                img_norm(inROI) = 0.5; 
            else
                val = (img(inROI) - min_val) / (max_val - min_val); 
                val(val < 0) = 0; 
                val(val > 1) = 1; 
                img_norm(inROI) = val; 
            end
        end
        
        function [S2REP, REP_QA] = calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets)
            B4_val = double(B4 * 10000) * scale_factors(1) + offsets(1); 
            B5_val = double(B5 * 10000) * scale_factors(2) + offsets(2);
            B6_val = double(B6 * 10000) * scale_factors(3) + offsets(3); 
            B7_val = double(B7 * 10000) * scale_factors(4) + offsets(4);
            
            invalid_reflect = (B4_val < 0 | B4_val > 1) | ...
                              (B5_val < 0 | B5_val > 1) | ...
                              (B6_val < 0 | B6_val > 1) | ...
                              (B7_val < 0 | B7_val > 1) | ...
                              isnan(B4_val) | isnan(B5_val) | isnan(B6_val) | isnan(B7_val);
                              
            [H, W] = size(B4); 
            S2REP = nan(H, W); 
            REP_QA = zeros(H, W); 
            
            REP_QA(invalid_reflect) = 3; 
            valid_pixel = ~invalid_reflect;
            
            numerator = ((B4_val + B7_val) / 2) - B5_val; 
            denominator = (B6_val - B5_val) + 1e-8;
            
            zero_denominator = valid_pixel & (abs(denominator) < 1e-6); 
            REP_QA(zero_denominator) = 2; 
            valid_pixel(zero_denominator) = false;
            
            S2REP(valid_pixel) = 705 + 35 * (numerator(valid_pixel) ./ denominator(valid_pixel));
            
            rep_out_range = valid_pixel & (S2REP < 680 | S2REP > 760); 
            REP_QA(rep_out_range) = 4; 
            S2REP(rep_out_range) = NaN; 
            REP_QA(valid_pixel & ~rep_out_range) = 1;
        end
        
        %% ================= 5. [核心更新] 本征吸收光谱逻辑 =================
        function F_abs = computeIntrinsicAbsorption(ast, mineral_type)
            % ASTER波段：1(0.56um),2(0.66um),3N(0.81um),4(1.6um),5(2.1um),6(2.2um),7(2.3um),8(2.5um)
            eps_val = 1e-6;
            [H,W] = size(ast(:,:,1));
            F_abs = nan(H,W,'single'); % 初始化为NaN，非0
            
            switch lower(mineral_type)
                case 'gold' % 黄铁矿(Fe-S:0.8-0.9um) + Al-OH(2.2um)
                    cont = (ast(:,:,3) + ast(:,:,5))/2; % continuum
                    target = ast(:,:,3); % Fe-S吸收
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.5 * (ast(:,:,6) ./ (ast(:,:,5) + eps_val)); % 补充Al-OH吸收
                
                case 'copper' % Cu²⁺(0.8-0.9um) + OH(2.2um)
                    cont = (ast(:,:,3) + ast(:,:,5))/2;
                    target = ast(:,:,3);
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.5 * (ast(:,:,6) ./ (ast(:,:,5) + eps_val));
                
                case 'iron' % 铁氧化物(0.8um)
                    cont = (ast(:,:,2) + ast(:,:,4))/2;
                    target = ast(:,:,3);
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                case 'coal' % 有机质(2.3um)
                    cont = (ast(:,:,5) + ast(:,:,8))/2;
                    target = ast(:,:,7);
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                case 'rare_earth' % REE电子跃迁(2.2um)
                    cont = (ast(:,:,5) + ast(:,:,7))/2;
                    target = ast(:,:,6);
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                % ==================== 补充矿种计算逻辑 ====================
                case 'silver' % 方铅矿(Pb-S:1.0um) + OH(2.2um)
                    cont = (ast(:,:,3) + ast(:,:,4))/2; % 连续谱（0.81+1.6um）
                    target = ast(:,:,3); % Pb-S弱吸收（0.81um近1.0um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.4 * (ast(:,:,6) ./ (ast(:,:,5) + eps_val)); % 补充OH吸收
                
                case 'aluminum' % 高岭石(Al-OH:2.2um)
                    cont = (ast(:,:,5) + ast(:,:,7))/2; % 连续谱（2.1+2.3um）
                    target = ast(:,:,6); % Al-OH吸收（2.2um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.3 * (ast(:,:,4) ./ (ast(:,:,5) + eps_val)); % 补充1.6um基线
                
                case 'lead' % 方铅矿(Pb-S:1.0um)
                    cont = (ast(:,:,3) + ast(:,:,4))/2; % 连续谱（0.81+1.6um）
                    target = ast(:,:,3); % Pb-S弱吸收（0.81um近1.0um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                case 'zinc' % 闪锌矿(Zn-Fe:0.9-1.1um)
                    cont = (ast(:,:,3) + ast(:,:,4))/2; % 连续谱（0.81+1.6um）
                    target = ast(:,:,3); % Zn-Fe吸收（0.81um近0.9um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.2 * (ast(:,:,6) ./ (ast(:,:,7) + eps_val)); % 补充粘土伴生
                
                case 'nickel' % 硅镁镍矿(Ni-OH/Mg-OH:1.8-2.3um)
                    cont = (ast(:,:,5) + ast(:,:,7))/2; % 连续谱（2.1+2.3um）
                    target = ast(:,:,6); % Ni-OH吸收（2.2um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.3 * (ast(:,:,4) ./ (ast(:,:,5) + eps_val)); % 补充1.6um基线
                
                case 'cobalt' % 异极矿(Co²⁺:0.5-0.6um)
                    cont = (ast(:,:,1) + ast(:,:,2))/2; % 连续谱（0.56+0.66um）
                    target = ast(:,:,1); % Co²⁺电子跃迁（0.56um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                case 'molybdenum' % 辉钼矿(Fe相关:0.9um)
                    cont = (ast(:,:,2) + ast(:,:,3))/2; % 连续谱（0.66+0.81um）
                    target = ast(:,:,3); % Fe相关吸收（0.81um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                case 'fluorite' % 萤石(弱OH:1.4um，ASTER无1.4um用2.2um替代)
                    cont = (ast(:,:,5) + ast(:,:,7))/2; % 连续谱（2.1+2.3um）
                    target = ast(:,:,6); % 弱OH吸收（2.2um）
                    F_abs = (cont - target) ./ (cont + eps_val) * 0.5; % 萤石吸收弱，权重降低
                
                case 'tin' % 锡石(Sn-Fe:1.0um + OH:2.2um)
                    cont = (ast(:,:,3) + ast(:,:,5))/2; % 连续谱（0.81+2.1um）
                    target = ast(:,:,3); % Sn-Fe吸收（0.81um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.4 * (ast(:,:,6) ./ (ast(:,:,5) + eps_val)); % 补充OH吸收
                
                case 'tungsten' % 黑钨矿(W-Fe:0.9-1.0um)
                    cont = (ast(:,:,2) + ast(:,:,3))/2; % 连续谱（0.66+0.81um）
                    target = ast(:,:,3); % W-Fe吸收（0.81um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                case 'petroleum' % 石油(C-H:1.7-1.75um + 2.3-2.5um)
                    cont = (ast(:,:,4) + ast(:,:,8))/2; % 连续谱（1.6+2.5um）
                    target = ast(:,:,7); % C-H组合吸收（2.3um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.3 * (ast(:,:,4) ./ (ast(:,:,5) + eps_val)); % 补充1.7um近1.6um
                
                case 'gas' % 天然气(CH₄:1.65-1.7um + 2.3um)
                    cont = (ast(:,:,4) + ast(:,:,7))/2; % 连续谱（1.6+2.3um）
                    target = ast(:,:,4); % CH₄吸收（1.6um近1.65um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                case 'coalbed_gas' % 煤层气(C-H:1.7um + 2.3um)
                    cont = (ast(:,:,5) + ast(:,:,8))/2; % 连续谱（2.1+2.5um）
                    target = ast(:,:,7); % C-H吸收（2.3um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.2 * (ast(:,:,4) ./ (ast(:,:,5) + eps_val)); % 补充1.7um
                
                case 'helium' % 氦气(无直接吸收，用围岩OH替代)
                    cont = (ast(:,:,5) + ast(:,:,7))/2; % 连续谱（2.1+2.3um）
                    target = ast(:,:,6); % OH吸收（2.2um）
                    F_abs = (cont - target) ./ (cont + eps_val) * 0.2; % 权重极低
                
                case 'lithium' % 锂辉石(Li-OH/Al-OH:2.2-2.4um)
                    cont = (ast(:,:,6) + ast(:,:,8))/2; % 连续谱（2.2+2.5um）
                    target = ast(:,:,7); % Li-OH吸收（2.3um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.3 * (ast(:,:,6) ./ (ast(:,:,7) + eps_val)); % 补充2.2um
                
                case 'natural_hydrogen' % 自然氢气(无直接吸收，用围岩Fe替代)
                    cont = (ast(:,:,2) + ast(:,:,3))/2; % 连续谱（0.66+0.81um）
                    target = ast(:,:,3); % Fe吸收（0.81um）
                    F_abs = (cont - target) ./ (cont + eps_val) * 0.2; % 权重极低
                
                case 'potassium' % 钾长石(弱OH:1.4um，用2.2um替代)
                    cont = (ast(:,:,5) + ast(:,:,7))/2; % 连续谱（2.1+2.3um）
                    target = ast(:,:,6); % 弱OH吸收（2.2um）
                    F_abs = (cont - target) ./ (cont + eps_val) * 0.3; % 权重降低
                
                case 'uranium' % 沥青铀矿(U⁴⁺/U⁶⁺:0.8-1.0um)
                    cont = (ast(:,:,2) + ast(:,:,4))/2; % 连续谱（0.66+1.6um）
                    target = ast(:,:,3); % U氧化电子跃迁（0.81um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                
                case 'cave' % 洞穴(无矿物吸收，用DEM衍生指数替代，此处返回NaN)
                    F_abs = nan(H,W,'single');
                
                case 'offshore_petroleum' % 海底石油(同石油，增强2.5um权重)
                    cont = (ast(:,:,4) + ast(:,:,8))/2; % 连续谱（1.6+2.5um）
                    target = ast(:,:,7); % C-H组合吸收（2.3um）
                    F_abs = (cont - target) ./ (cont + eps_val);
                    F_abs = F_abs + 0.4 * (ast(:,:,8) ./ (ast(:,:,7) + eps_val)); % 增强2.5um
                
                % ==================== 原有通用模式 ====================
                otherwise % 通用模式：OH吸收(2.2um)
                    cont = (ast(:,:,5) + ast(:,:,7))/2;
                    target = ast(:,:,6);
                    F_abs = (cont - target) ./ (cont + eps_val);
            end
            
            F_abs(isinf(F_abs)) = NaN;
        end
        
        function indices = computeDEMIndices(dem, mineral_type, H, W, inROI)
             indices = struct('slope', nan(H,W,'single'), 'neg_curvature', nan(H,W,'single'));
             
             if ~strcmpi(mineral_type, 'cave'), return; end
             
             [dx, dy] = gradient(dem); 
             slope = atan(sqrt(dx.^2 + dy.^2)) * 180 / pi; 
             indices.slope = GeoUtils.mat2gray_roi(slope, inROI); 
             
             [dxx, ~] = gradient(dx); 
             [~, dyy] = gradient(dy); 
             curvature = -(dxx + dyy); 
             neg_curv = max(-curvature, 0); 
             indices.neg_curvature = GeoUtils.mat2gray_roi(neg_curv, inROI); 
        end
        
        function local_sum = calc_local_sum_with_nan(Z)
            % 计算局部邻域和 (忽略 NaN)
            [rows, cols] = size(Z); 
            pad = 1; 
            Z_padded = padarray(Z, [pad, pad], NaN, 'both'); 
            local_sum = nan(rows, cols); 
            
            % 3x3 模板，不包含中心点
            w = ones(3,3); 
            w(2,2) = 0;
            
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
            local_sum(isinf(local_sum)) = NaN;
        end
        
        %% ================= 6. [核心更新] 矿物阈值与增强公式 =================
        function [F_thr, delta_thr, Moran_thr, enh_func] = getMineralThresholds(mineral_type)
            switch lower(mineral_type)
                case 'gold'
                    F_thr = 0.018; delta_thr = -2; Moran_thr = 0.20;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.45*Ferric + 0.25*Fe_anomaly + 0.15*Hydroxy_anomaly + 0.10*Clay + 0.05*NDVI_inv;
                
                case 'copper'
                    F_thr = 0.020; delta_thr = -3; Moran_thr = 0.25;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.40*Clay + 0.30*Hydroxy_anomaly + 0.20*Ferric + 0.10*Fe_anomaly;
                
                case 'iron'
                    F_thr = 0.030; delta_thr = -4; Moran_thr = 0.35;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.60*Ferric + 0.40*Fe_anomaly;
                
                case 'lead'
                    F_thr = 0.025; delta_thr = -3; Moran_thr = 0.30;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.40*Hydroxy_anomaly + 0.30*Clay + 0.30*Ferric;
                
                case 'zinc'
                    F_thr = 0.024; delta_thr = -3; Moran_thr = 0.28;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.40*Hydroxy_anomaly + 0.30*Clay + 0.30*Ferric;
                
                case 'molybdenum'
                    F_thr = 0.028; delta_thr = -4; Moran_thr = 0.32;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.50*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric;
                
                case 'copper_gold'
                    F_thr = 0.019; delta_thr = -2.5; Moran_thr = 0.22;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.40*Ferric + 0.40*Clay + 0.20*Hydroxy_anomaly;
                
                case 'coal'
                    F_thr = 0.032; delta_thr = -4.5; Moran_thr = 0.38;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.60*NDVI_inv + 0.40*Hydroxy_anomaly;
                
                case 'tin'
                    F_thr = 0.023; delta_thr = -2.5; Moran_thr = 0.26;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.50*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric;
                
                case 'petroleum'
                    F_thr = 0.035; delta_thr = -5; Moran_thr = 0.40;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.70*NDVI_inv + 0.30*Hydroxy_anomaly;
                
                case 'gas'
                    F_thr = 0.033; delta_thr = -4.5; Moran_thr = 0.38;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.60*NDVI_inv + 0.40*Hydroxy_anomaly;
                
                case 'lithium'
                    F_thr = 0.022; delta_thr = -2; Moran_thr = 0.25;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.60*Clay + 0.40*Hydroxy_anomaly;
                
                case 'nickel'
                    F_thr = 0.026; delta_thr = -3.5; Moran_thr = 0.30;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.50*Hydroxy_anomaly + 0.30*Ferric + 0.20*Clay;
                
                case 'fluorite'
                    F_thr = 0.029; delta_thr = -4; Moran_thr = 0.35;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.50*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric;
                
                case 'phosphate'
                    F_thr = 0.027; delta_thr = -3.5; Moran_thr = 0.32;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.40*Hydroxy_anomaly + 0.30*Clay + 0.30*NDVI_inv;
                
                case 'rare_earth'
                    F_thr = 0.026; delta_thr = -3; Moran_thr = 0.28;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.40*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric + 0.10*NDVI_inv;
                
                case 'helium'
                    F_thr = 0.031; delta_thr = -4; Moran_thr = 0.36;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.50*NDVI_inv + 0.30*Hydroxy_anomaly + 0.20*Clay;
                
                case 'uranium'
                    F_thr = 0.028; delta_thr = -4.5; Moran_thr = 0.32;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.40*Fe_anomaly + 0.30*Ferric + 0.20*NDVI_inv + 0.10*Hydroxy_anomaly;
                
                case 'natural_hydrogen'
                    F_thr = 0.032; delta_thr = -4; Moran_thr = 0.37;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.50*NDVI_inv + 0.30*Clay + 0.20*Hydroxy_anomaly;
                
                case 'potassium'
                    F_thr = 0.025; delta_thr = -3; Moran_thr = 0.28;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.40*Hydroxy_anomaly + 0.30*Clay + 0.20*Ferric + 0.10*NDVI_inv;
                
                case 'cave'
                    F_thr = 0.025; delta_thr = -3; Moran_thr = 0.30;
                    % 洞穴模式多slope和curvature，同样移除内部inROI
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv, slope, curvature) ...
                        0.30*NDVI_inv + 0.25*slope + 0.20*curvature + 0.15*Hydroxy_anomaly + 0.10*Clay;
                
                case 'offshore_petroleum'
                    F_thr = 0.030; delta_thr = -4; Moran_thr = 0.35;
                    % 海底石油模式多OSI和SDS，移除内部inROI
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv, OSI, SDS) ...
                        0.40*OSI + 0.30*SDS + 0.20*NDVI_inv + 0.10*Hydroxy_anomaly;
                
                otherwise
                    F_thr = 0.018; delta_thr = -2; Moran_thr = 0.20;
                    enh_func = @(Ferric, Fe_anomaly, Hydroxy_anomaly, Clay, NDVI_inv) ...
                        0.45*Ferric + 0.25*Fe_anomaly + 0.15*Hydroxy_anomaly + 0.10*Clay + 0.05*NDVI_inv;
                    warning('未知矿种，使用默认金矿阈值');
            end
        end
        
        %% ================= 7. [核心更新] 共振参数表 =================
        function params = getYakymchukParams(mineral_type)
            param_table = struct(...
                'gold',             struct('a',50, 'b',150, 'c',20), ...
                'silver',           struct('a',45, 'b',135, 'c',19), ...
                'copper',           struct('a',40, 'b',120, 'c',18), ...
                'iron',             struct('a',35, 'b',100, 'c',15), ...
                'aluminum',         struct('a',48, 'b',140, 'c',19), ...
                'coal',             struct('a',32, 'b',80, 'c',16), ...
                'lead',             struct('a',42, 'b',125, 'c',18), ...
                'zinc',             struct('a',42, 'b',125, 'c',18), ...
                'nickel',           struct('a',35, 'b',105, 'c',16), ...
                'cobalt',           struct('a',38, 'b',115, 'c',17), ...
                'molybdenum',       struct('a',48, 'b',140, 'c',20), ...
                'rare_earth',       struct('a',45, 'b',140, 'c',18), ...
                'fluorite',         struct('a',55, 'b',170, 'c',22), ...
                'tin',              struct('a',52, 'b',155, 'c',21), ...
                'tungsten',         struct('a',52, 'b',155, 'c',21), ...
                'petroleum',        struct('a',30, 'b',70, 'c',15), ...
                'gas',              struct('a',28, 'b',75, 'c',14), ...
                'coalbed_gas',      struct('a',32, 'b',80, 'c',16), ...
                'helium',           struct('a',25, 'b',85, 'c',14), ...
                'lithium',          struct('a',40, 'b',110, 'c',17), ...
                'natural_hydrogen', struct('a',30, 'b',80, 'c',15), ...
                'potassium',        struct('a',45, 'b',135, 'c',19), ...
                'uranium',          struct('a',40, 'b',130, 'c',19), ...
                'cave',             struct('a',40, 'b',120, 'c',18));
            
            if isfield(param_table, lower(mineral_type))
                params = param_table.(lower(mineral_type));
            else
                params = struct('a',50, 'b',150, 'c',20);
                warning('未知矿种，使用默认金矿参数');
            end
        end
    end
end