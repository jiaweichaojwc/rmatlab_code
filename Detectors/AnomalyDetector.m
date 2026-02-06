classdef (Abstract) AnomalyDetector < handle
    methods (Abstract)
        % 必须返回 struct('mask', binaryMatrix, 'debug', struct(...))
        result = calculate(obj, dataCtx); 
    end
end