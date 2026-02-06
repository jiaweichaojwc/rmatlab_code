classdef FusionEngine < handle
    properties
        detectors % Map: Name -> DetectorObject
        results   % Map: Name -> ResultStruct (Mask, DebugData)
    end
    
    methods
        function obj = FusionEngine()
            obj.detectors = containers.Map();
            obj.results = containers.Map();
        end
        
        function addDetector(obj, name, detectorObj)
            obj.detectors(name) = detectorObj;
        end
        
        function computeAll(obj, dataCtx)
            keys = obj.detectors.keys;
            for i = 1:length(keys)
                name = keys{i};
                fprintf('  -> 计算探测器: %s ...\n', name);
                detector = obj.detectors(name);
                % 调用具体探测器的 calculate 方法
                obj.results(name) = detector.calculate(dataCtx);
            end
        end
        
        function fusedMask = getFusedMask(obj, namesList)
            % 灵活融合：传入 {'RedEdge', 'Intrinsic'} 即可只融合这两个
            if ischar(namesList), namesList = {namesList}; end
            
            fusedMask = [];
            for i = 1:length(namesList)
                name = namesList{i};
                if ~obj.results.isKey(name)
                    error('探测器 %s 尚未计算或未注册', name);
                end
                
                res = obj.results(name);
                currentMask = res.mask;
                
                % 统一尺寸
                if isempty(fusedMask)
                    fusedMask = currentMask;
                else
                    if ~isequal(size(fusedMask), size(currentMask))
                        currentMask = imresize(currentMask, size(fusedMask), 'nearest');
                    end
                    % OR 逻辑融合
                    fusedMask = fusedMask | currentMask;
                end
            end
            fusedMask = double(fusedMask);
            fprintf('>>> 融合完成，包含了: %s\n', strjoin(namesList, ', '));
        end
        
        function res = getResult(obj, name)
            res = obj.results(name);
        end
    end
end