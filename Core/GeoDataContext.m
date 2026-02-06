classdef GeoDataContext < handle
    properties
        % 原始配置
        mineral_type
        region_type
        
        % 基础数据 (H x W x Bands)
        s2      % Sentinel-2 (Reflectance)
        lan     % Landsat-8
        ast     % ASTER
        dem     % DEM
        
        % 辅助数据
        inROI   % ROI Mask (已翻转)
        R       % 地理参考
        lonGrid, latGrid
        lonROI, latROI
        belt_lon, belt_lat
        
        % 预计算的常用波段 (懒加载或预加载)
        NIR, Red, Green, Blue
    end
    
    methods
        function obj = GeoDataContext(config)
            obj.mineral_type = config.mineral_type;
            obj.region_type = config.region_type;
            
            % --- 1. 获取路径 (将原 switch-case 封装进 helper) ---
            [data_dir, roi_file, obj.belt_lon, obj.belt_lat] = GeoUtils.getRegionConfig(obj.region_type);
            
            % --- 2. 读取数据 (使用 GeoUtils 静态方法) ---
            [obj.s2, obj.R] = GeoUtils.readSentinel2(data_dir);
            obj.lan = GeoUtils.readLandsat8(data_dir, obj.R);
            obj.ast = GeoUtils.readASTER(data_dir, obj.R);
            [obj.dem, obj.inROI, obj.lonGrid, obj.latGrid, obj.lonROI, obj.latROI] = ...
                GeoUtils.readDEMandROI(data_dir, roi_file, obj.R);
            
            % --- 3. 基础波段提取 ---
            obj.NIR   = GeoUtils.getBand(obj.s2, obj.lan, 4);
            obj.Red   = GeoUtils.getBand(obj.s2, obj.lan, 3);
            obj.Green = GeoUtils.getBand(obj.s2, obj.lan, 2);
            obj.Blue  = GeoUtils.getBand(obj.s2, obj.lan, 1);
            
            % 填充 ASTER 的 NaN
            obj.fillAsterNaN();
        end
        
        function fillAsterNaN(obj)
            % 仅填充 ROI 内的 NaN
            for b = 1:size(obj.ast, 3)
                bandData = obj.ast(:,:,b);
                roi_vals = bandData(obj.inROI);
                mean_val = mean(roi_vals, 'omitnan');
                if ~isnan(mean_val)
                    mask = obj.inROI & isnan(bandData);
                    bandData(mask) = mean_val;
                    obj.ast(:,:,b) = bandData;
                end
            end
        end
    end
end