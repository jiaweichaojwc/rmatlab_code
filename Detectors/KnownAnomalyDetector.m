classdef KnownAnomalyDetector < AnomalyDetector
    methods
        function res = calculate(obj, ctx)
            fprintf('  [KnownAnomaly] 正在处理 KML 已知异常...\n');
            
            % 1. 检查配置
            if isempty(ctx.kmz_path) || ~exist(ctx.kmz_path, 'file')
                fprintf('    ⚠️ KML文件不存在或未选择，跳过。\n');
                res.mask = zeros(size(ctx.inROI));
                res.debug.raw = [];
                return;
            end
            
            if isempty(ctx.ref_tif_path)
                warning('    ❌ 缺少参考影像路径(ref_tif_path)，无法校正KML坐标。');
                res.mask = zeros(size(ctx.inROI));
                return;
            end
            
            % 2. 调用 KMZMaskGenerator (对应 untitled3.m Step 3.5)
            try
                radius = 3; % 默认扩充半径
                % 实例化生成器
                generator = KMZMaskGenerator(ctx.kmz_path, ctx.ref_tif_path, ctx.kmz_keywords, radius);
                
                % 运行生成
                raw_mask = generator.run();
                
                % 3. 尺寸对齐
                targetSize = size(ctx.inROI);
                if ~isequal(size(raw_mask), targetSize)
                    fprintf('    ⚠️ 尺寸不匹配 (%dx%d vs %dx%d)，调整中...\n', ...
                        size(raw_mask,1), size(raw_mask,2), targetSize(1), targetSize(2));
                    raw_mask = imresize(raw_mask, targetSize, 'nearest');
                end
                
                % 4. 结果封装
                mask = double(raw_mask);
                mask(~ctx.inROI) = 0; % 裁剪到 ROI
                
                res.mask = mask;
                res.debug.raw = raw_mask;
                fprintf('    ✅ KML异常提取完成，像素数: %d\n', sum(mask(:)>0));
                
            catch ME
                fprintf('    ❌ KML处理出错: %s\n', ME.message);
                res.mask = zeros(size(ctx.inROI));
            end
        end
    end
end