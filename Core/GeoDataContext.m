classdef GeoDataContext < handle
    properties
        % 配置
        mineral_type
        region_type
        levashov_mode = true
        
        % [新增] KML 配置
        kmz_path
        kmz_keywords
        
        % 路径信息
        data_dir 
        ref_tif_path % [新增] 用于 KML 地理校正的参考影像路径
        
        % 数据
        s2, lan, ast, dem
        inROI, R, lonGrid, latGrid, lonROI, latROI
        belt_lon, belt_lat
        NIR, Red, Green, Blue
    end
    
    methods
        function obj = GeoDataContext(config)
            obj.mineral_type = config.mineral_type;
            obj.region_type = config.region_type;
            if isfield(config, 'levashov_mode'), obj.levashov_mode = config.levashov_mode; end
            
            % 接收 KML 配置
            if isfield(config, 'kmz_path'), obj.kmz_path = config.kmz_path; end
            if isfield(config, 'kmz_keywords'), obj.kmz_keywords = config.kmz_keywords; end
            
            % 1. 获取路径 (交互/自动)
            [data_dir_local, roi_file, obj.belt_lon, obj.belt_lat] = GeoUtils.getRegionConfig(obj.region_type);
            obj.data_dir = data_dir_local;
            
            % 2. 读取数据
            fprintf('  [Context] Loading data from: %s\n', obj.data_dir);
            
            % [修改] 接收 ref_tif_path (untitled3.m 需要用它来做 KML 的基准)
            [obj.s2, obj.R, obj.ref_tif_path] = GeoUtils.readSentinel2(obj.data_dir);
            
            obj.lan = GeoUtils.readLandsat8(obj.data_dir, obj.R);
            obj.ast = GeoUtils.readASTER(obj.data_dir, obj.R);
            [obj.dem, obj.inROI, obj.lonGrid, obj.latGrid, obj.lonROI, obj.latROI] = ...
                GeoUtils.readDEMandROI(obj.data_dir, roi_file, obj.R);
            
            obj.NIR   = GeoUtils.getBand(obj.s2, obj.lan, 4);
            obj.Red   = GeoUtils.getBand(obj.s2, obj.lan, 3);
            obj.Green = GeoUtils.getBand(obj.s2, obj.lan, 2);
            obj.Blue  = GeoUtils.getBand(obj.s2, obj.lan, 1);
            
            obj.fillAsterNaN();
        end
        
        function fillAsterNaN(obj)
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