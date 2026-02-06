classdef GeoDataContext < handle
    properties
        % 原始配置
        mineral_type
        region_type
        levashov_mode = true
        
        % 【关键修改】新增 data_dir 属性，以便外部调用
        data_dir 
        
        % 基础数据
        s2, lan, ast, dem
        
        % 辅助数据
        inROI, R, lonGrid, latGrid, lonROI, latROI
        belt_lon, belt_lat
        
        % 常用波段
        NIR, Red, Green, Blue
    end
    
    methods
        function obj = GeoDataContext(config)
            obj.mineral_type = config.mineral_type;
            obj.region_type = config.region_type;
            if isfield(config, 'levashov_mode')
                obj.levashov_mode = config.levashov_mode;
            end
            
            % --- 1. 获取路径 (封装在 GeoUtils) ---
            % 这里会触发弹窗（如果是手动模式）
            [data_dir_local, roi_file, obj.belt_lon, obj.belt_lat] = GeoUtils.getRegionConfig(obj.region_type);
            
            % 【关键修改】把局部变量存入对象属性
            obj.data_dir = data_dir_local;
            
            % --- 2. 读取数据 (使用 obj.data_dir) ---
            fprintf('  [Context] Loading data from: %s\n', obj.data_dir);
            [obj.s2, obj.R] = GeoUtils.readSentinel2(obj.data_dir);
            obj.lan = GeoUtils.readLandsat8(obj.data_dir, obj.R);
            obj.ast = GeoUtils.readASTER(obj.data_dir, obj.R);
            [obj.dem, obj.inROI, obj.lonGrid, obj.latGrid, obj.lonROI, obj.latROI] = ...
                GeoUtils.readDEMandROI(obj.data_dir, roi_file, obj.R);
            
            % --- 3. 基础波段提取 ---
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