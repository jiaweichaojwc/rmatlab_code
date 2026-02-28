function result_mat_path = run_core_algorithm(data_dir, roi_file, mineral_type, kmz_path, kmz_threshold)
    % run_core_algorithm - 供 Python 调用的核心计算引擎接口
    
    fprintf('\n=== 舒曼波共振遥感核心引擎启动 ===\n');
    fprintf('目标矿种: %s\n', mineral_type);
    
    % 1. 配置参数装载
    config.mineral_type = char(mineral_type);
    config.region_type  = ''; 
    config.levashov_mode = true;
    config.fusion_mode = true;
    config.kmz_threshold = double(kmz_threshold);
    
    % 处理 KML 路径 (如果是空字符串说明不启用)
    kmz_path_str = char(kmz_path);
    if strlength(kmz_path_str) > 0
        config.kmz_path = kmz_path_str;
        config.kmz_keywords = {'矿体投影', 'Object ID', 'ZK', '异常', '已知矿点'}; 
    else
        config.kmz_path = '';
    end
    
    config.data_dir = char(data_dir);
    config.roi_file = char(roi_file);
    
    % 2. 动态确定使用的探测器并构造输出文件夹
    detectors_to_use = {'RedEdge', 'Intrinsic', 'SlowVars'};
    if ~isempty(config.kmz_path)
        detectors_to_use{end+1} = 'KnownAnomaly';
    end
    
    types_str = strjoin(detectors_to_use, '_');
    folder_name = [types_str, '_Result_', config.mineral_type, '_', datestr(now, 'yyyymmdd_HHMM')];
    
    % 初始化数据上下文
    dataCtx = GeoDataContext(config);
    
    % 确定最终输出目录
    config.outDir = fullfile(dataCtx.data_dir, folder_name);
    if ~exist(config.outDir, 'dir')
        mkdir(config.outDir);
    end
    
    % 3. 构建并运行融合引擎
    engine = FusionEngine();
    engine.addDetector('RedEdge', RedEdgeDetector());
    engine.addDetector('Intrinsic', IntrinsicDetector());
    engine.addDetector('SlowVars', SlowVarsDetector());
    if ~isempty(config.kmz_path)
        engine.addDetector('KnownAnomaly', KnownAnomalyDetector());
    end
    
    fprintf('>>> 开始并行计算各异常层...\n');
    engine.computeAll(dataCtx);
    
    fprintf('>>> 执行特征融合...\n');
    final_mask = engine.getFusedMask(detectors_to_use);
    
    % 4. 后处理并生成 .mat 结果
    fprintf('>>> 执行深度与压力反演及后处理...\n');
    PostProcessor.run(dataCtx, engine, final_mask, config.outDir);
    
    % 5. 组装结果文件路径返回给 Python
    % PostProcessor.m 中保存的文件名规则是: [mineral_type]_Result.mat
    result_mat_path = fullfile(config.outDir, sprintf('%s_Result.mat', config.mineral_type));
    fprintf('=== 核心计算完成 ===\n');
end