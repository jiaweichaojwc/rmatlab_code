%{
% =========================================================================
%   主程序：调用 KMZMaskGenerator 生成蒙版
% =========================================================================
clear; clc;

% 1. 设置文件路径 (已更新为你提供的路径)
kmzPath = 'C:\Users\Administrator\Desktop\5000异常-全.kml';

tifPath = 'C:\Deep-Explor\data\WXWork\1688856523802776\WeDrive\深层探索\遥感数据\下载任务结果\新疆高昌区库格孜觉北金矿-59.05km2【金】（四川黄金）（20260104任务，20260105下载）\data\Sentinel 2 L2\B08.tiff';

outputBase = 'C:\Users\Administrator\Desktop\mask_result';

% 2. 设置参数
% 关键词：只要KML里的名字包含这些词，就会被提取并叠加到蒙版中
% 例如 "Object ID 14" 包含 "Object ID"，会被自动提取
targetKeywords = {'矿体投影', 'Object ID', 'ZK', '异常'}; 

% 单点扩充半径 (如果KML里有点状地标，会扩充为 3px 的圆点)
pointRadius = 3; 

% 3. 执行处理
try
    fprintf('🚀 开始运行...\n');
    
    % 初始化生成器
    generator = KMZMaskGenerator(kmzPath, tifPath, targetKeywords, pointRadius);
    generator.OutputBase = outputBase;
    
    % 运行并获取蒙版矩阵 (逻辑值 0和1)
    mask = generator.run();
    
    % 4. 简单的结果展示
    figure;
    imshow(mask);
    title('生成的蒙版结果 (白色为目标区域)');
    
    fprintf('\n✅ 运行成功！\n');
    fprintf('   结果保存在: %s.mat 和 %s_VisualCheck.png\n', outputBase, outputBase);
    
catch ME
    fprintf('\n❌ 运行出错: %s\n', ME.message);
    % 如果是路径太长导致的问题，MATLAB可能会报错，请检查路径是否正确
end
%}
classdef KMZMaskGenerator
    % KMZMaskGenerator 遥感影像蒙版生成器 (最终修复版)
    
    properties
        KmzPath
        TifPath
        TargetKeywords
        PointRadiusPixel
        OutputBase
        
        % --- 内部状态变量 ---
        TifGeoInfo      % SpatialRef 对象
        TifSize         % [Height, Width]
        TifLimits       % struct('xMin',..., 'yMax',...) 用于存储边界
        IsGeographic    % Boolean: 是否为地理坐标系(经纬度)
        Mask            % 最终的二值蒙版
        GeoData         % <--- 【关键修复】补回了这个属性
    end
    
    methods
        function obj = KMZMaskGenerator(kmzPath, tifPath, targetKeywords, pointRadiusPixel)
            obj.KmzPath = kmzPath;
            obj.TifPath = tifPath;
            if nargin < 3 || isempty(targetKeywords)
                obj.TargetKeywords = {'矿体投影', 'Object ID', 'ZK'};
            else
                obj.TargetKeywords = targetKeywords;
            end
            if nargin < 4
                obj.PointRadiusPixel = 3;
            else
                obj.PointRadiusPixel = pointRadiusPixel;
            end
            
            % 初始化结构体
            obj.GeoData = struct('type', {}, 'name', {}, 'coords', {});
        end
        
        function obj = extractAndParseKml(obj)
            fprintf('\n🔄 正在读取 KML 数据...\n');
            [~, ~, ext] = fileparts(obj.KmzPath);
            kmlContent = '';
            tempDir = '';
            
            try
                if strcmpi(ext, '.kmz') || strcmpi(ext, '.ovkmz')
                    tempDir = tempname;
                    unzip(obj.KmzPath, tempDir);
                    files = dir(fullfile(tempDir, '*.kml'));
                    if isempty(files), error('KMZ中未找到KML文件'); end
                    kmlContent = fileread(fullfile(files(1).folder, files(1).name));
                else
                    kmlContent = fileread(obj.KmzPath);
                end
                if ~isempty(tempDir) && exist(tempDir, 'dir'), rmdir(tempDir, 's'); end
            catch ME
                if ~isempty(tempDir) && exist(tempDir, 'dir'), rmdir(tempDir, 's'); end
                rethrow(ME);
            end
            
            fprintf('🔍 解析 KML...\n');
            placemarkPattern = '(?s)<Placemark>(.*?)</Placemark>';
            placemarks = regexp(kmlContent, placemarkPattern, 'tokens');
            
            count = 0;
            for i = 1:length(placemarks)
                pmContent = placemarks{i}{1};
                nameToken = regexp(pmContent, '<name>(.*?)</name>', 'tokens');
                if isempty(nameToken), continue; end
                areaName = strtrim(nameToken{1}{1});
                
                isMatch = false;
                for k = 1:length(obj.TargetKeywords)
                    if contains(areaName, obj.TargetKeywords{k}), isMatch = true; break; end
                end
                if ~isMatch, continue; end
                
                coordsToken = regexp(pmContent, '<coordinates>(.*?)</coordinates>', 'tokens');
                if isempty(coordsToken), continue; end
                
                for c = 1:length(coordsToken)
                    rawData = sscanf(strrep(strtrim(coordsToken{c}{1}), ',', ' '), '%f');
                    if isempty(rawData), continue; end
                    
                    if mod(length(rawData), 3) == 0
                        coords = reshape(rawData, 3, [])'; coords = coords(:, 1:2);
                    elseif mod(length(rawData), 2) == 0
                        coords = reshape(rawData, 2, [])';
                    else
                        continue;
                    end
                    
                    if contains(pmContent, '<Polygon>') || contains(pmContent, '<LinearRing>')
                        obj.GeoData(end+1) = struct('type', 'Polygon', 'name', areaName, 'coords', coords);
                        count = count + 1;
                        fprintf('  ✅ 匹配(多边形): %s\n', areaName);
                    elseif contains(pmContent, '<Point>')
                        obj.GeoData(end+1) = struct('type', 'Point', 'name', areaName, 'coords', coords(1,:));
                        count = count + 1;
                        fprintf('  ✅ 匹配(单点): %s\n', areaName);
                    end
                end
            end
            fprintf('   已提取 %d 个目标区域\n', count);
        end
        
        function obj = readTiffInfo(obj)
            fprintf('\n🖼️ 读取 TIFF 地理信息...\n');
            info = geotiffinfo(obj.TifPath);
            R = info.SpatialRef;
            obj.TifGeoInfo = R;
            obj.TifSize = [info.Height, info.Width];
            
            % --- 兼容性判断 ---
            obj.IsGeographic = false;
            
            if isprop(R, 'LongitudeLimits')
                obj.IsGeographic = true;
                xLims = R.LongitudeLimits;
                yLims = R.LatitudeLimits;
            elseif isprop(R, 'XWorldLimits')
                xLims = R.XWorldLimits;
                yLims = R.YWorldLimits;
            elseif isprop(R, 'XLimWorld')
                xLims = R.XLimWorld;
                yLims = R.YLimWorld;
            else
                % 最后的兜底
                if isprop(R, 'XIntrinsicLimits')
                     % 如果完全无法读取地理范围，尝试用 intrinsic
                     warning('无法读取地理坐标范围，尝试使用像素范围（可能导致位置错误）');
                     xLims = [0, info.Width];
                     yLims = [0, info.Height];
                else
                     error('无法识别的空间参考属性');
                end
            end
            
            obj.TifLimits = struct();
            obj.TifLimits.xMin = xLims(1);
            obj.TifLimits.xMax = xLims(2);
            obj.TifLimits.yMin = yLims(1);
            obj.TifLimits.yMax = yLims(2);
            
            fprintf('   尺寸: %d x %d\n', obj.TifSize(1), obj.TifSize(2));
            fprintf('   X/Lon 范围: %.6f ~ %.6f\n', obj.TifLimits.xMin, obj.TifLimits.xMax);
            fprintf('   Y/Lat 范围: %.6f ~ %.6f\n', obj.TifLimits.yMin, obj.TifLimits.yMax);
        end
        
        function obj = generateMask(obj)
            fprintf('\n🎨 生成蒙版...\n');
            height = obj.TifSize(1);
            width = obj.TifSize(2);
            obj.Mask = false(height, width);
            
            R = obj.TifGeoInfo;
            
            % 计算像素宽/高 (用于Fallback手动计算)
            pixelWidth = (obj.TifLimits.xMax - obj.TifLimits.xMin) / width;
            pixelHeight = (obj.TifLimits.yMax - obj.TifLimits.yMin) / height; 
            
            for i = 1:length(obj.GeoData)
                item = obj.GeoData(i);
                x = item.coords(:,1);
                y = item.coords(:,2);
                
                rows = []; cols = [];
                try
                    % 优先尝试内置函数
                    if obj.IsGeographic
                        [rows, cols] = R.worldToDiscrete(x, y);
                    else
                        [rows, cols] = R.worldToDiscrete(x, y);
                    end
                catch
                    % 失败则手动计算
                    cols = ceil((x - obj.TifLimits.xMin) / pixelWidth);
                    rows = ceil((obj.TifLimits.yMax - y) / pixelHeight);
                end
                
                % 过滤无效坐标
                validIdx = rows >= 1 & rows <= height & cols >= 1 & cols <= width;
                
                if strcmp(item.type, 'Polygon')
                    if ~isempty(rows)
                        if rows(1) ~= rows(end) || cols(1) ~= cols(end)
                            rows(end+1) = rows(1); cols(end+1) = cols(1);
                        end
                        maskPart = poly2mask(cols, rows, height, width);
                        obj.Mask = obj.Mask | maskPart;
                    end
                elseif strcmp(item.type, 'Point')
                    if any(validIdx)
                        r = rows(1); c = cols(1);
                        if c >= 1 && c <= width && r >= 1 && r <= height
                            rad = obj.PointRadiusPixel;
                            [xx, yy] = meshgrid(-rad:rad, -rad:rad);
                            disk = (xx.^2 + yy.^2) <= rad^2;
                            rMin = max(1, r-rad); rMax = min(height, r+rad);
                            cMin = max(1, c-rad); cMax = min(width, c+rad);
                            drMin = 1+(rMin-(r-rad)); drMax = size(disk,1)-((r+rad)-rMax);
                            dcMin = 1+(cMin-(c-rad)); dcMax = size(disk,2)-((c+rad)-cMax);
                            obj.Mask(rMin:rMax, cMin:cMax) = obj.Mask(rMin:rMax, cMin:cMax) | disk(drMin:drMax, dcMin:dcMax);
                        end
                    end
                end
            end
        end
        
        function saveResults(obj, outputBase)
            fprintf('\n💾 保存结果...\n');
            finalMask = obj.Mask;
            % 自动创建文件夹
            outDir = fileparts(outputBase);
            if ~exist(outDir, 'dir')
                mkdir(outDir);
            end
            
            save([outputBase, '.mat'], 'finalMask');
            imwrite(finalMask, [outputBase, '_VisualCheck.png']);
            fprintf('   保存完毕: %s_VisualCheck.png\n', outputBase);
        end
        
        function mask = run(obj)
            obj = obj.extractAndParseKml();
            obj = obj.readTiffInfo();
            obj = obj.generateMask();
            if ~isempty(obj.OutputBase), obj.saveResults(obj.OutputBase); end
            mask = obj.Mask;
            fprintf('\n🎉 完成!\n');
        end
    end
end