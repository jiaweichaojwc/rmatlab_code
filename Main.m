%% ================= 舒曼波共振遥感 - 面向对象主程序 =================
% Main.m
% 对应 untitled3.m 逻辑：全交互式读取 + 4掩码融合(含KML) + 动态文件夹命名 + 融合模式选择
clear all; close all; clc;
addpath(genpath(pwd)); 

% ================= 1. 基础配置 =================
config.mineral_type = 'gold'; 

% 【关键】置为空，触发 GeoUtils 交互式选择数据文件夹
config.region_type  = ''; 

config.levashov_mode = true;
config.kmz_threshold = 0.6;

% ================= 1.5 [新增] 融合模式交互式选择 =================
fprintf('>>> [交互模式] 是否叠加地表背景 (综合融合模式)？\n');
fusion_ans = questdlg('是否叠加地表通用背景进行平滑融合？', ...
                      '融合模式配置', ...
                      '是(综合融合)', '否(纯净模式)', '是(综合融合)');

if strcmp(fusion_ans, '是(综合融合)')
    config.fusion_mode = true;
    fprintf('✅ 已选择：综合融合模式 (叠加地表背景并进行高斯平滑)\n');
else
    config.fusion_mode = false;
    fprintf('✅ 已选择：纯净模式 (直接输出探测器原始等值线)\n');
end

% ================= 2. KML 已知异常配置 (交互式选择) =================
fprintf('>>> [交互模式] 是否导入 KML/KMZ 已知异常文件 (第4掩码)？\n');
answer = questdlg('是否导入 KML/KMZ 已知异常文件？', 'KML配置', '是', '否', '是');

% 定义基础探测器列表
%'RedEdge', 'Intrinsic', 'SlowVars'
detectors_to_use = {'Intrinsic'};

if strcmp(answer, '是')
    [kml_file, kml_path] = uigetfile({'*.kml;*.kmz', 'Google Earth Files (*.kml, *.kmz)'}, '请选择已知异常文件');
    if kml_file ~= 0
        config.kmz_path = fullfile(kml_path, kml_file);
        fprintf('✅ 已选择 KML 文件: %s\n', config.kmz_path);
        
        % 如果选择了 KML，加入 KnownAnomaly 到列表
        detectors_to_use{end+1} = 'KnownAnomaly';
    else
        config.kmz_path = '';
        fprintf('⚠️ 用户取消选择 KML 文件，将跳过此步骤。\n');
    end
else
    config.kmz_path = '';
    fprintf('>>> 跳过 KML 导入。\n');
end

% KML 提取关键词
config.kmz_keywords = {'矿体投影', 'Object ID', 'ZK', '异常', '已知矿点'}; 

% ================= 3. 初始化数据上下文 =================
% 这里会触发 GeoUtils.getRegionConfig 弹窗选择数据文件夹 + 坐标文件
dataCtx = GeoDataContext(config);

% ================= 4. 设定输出路径 (动态命名) =================
% 将探测器类型拼接成字符串 (例如: RedEdge_Intrinsic_SlowVars_KnownAnomaly)
types_str = strjoin(detectors_to_use, '_');

% 构造文件夹名称
folder_name = [types_str, '_Result_', config.mineral_type, '_', datestr(now, 'yyyymmdd_HHMM')];
config.outDir = fullfile(dataCtx.data_dir, folder_name);

if ~exist(config.outDir, 'dir')
    mkdir(config.outDir);
end
fprintf('📂 结果将保存至: %s\n', config.outDir);

% ================= 5. 初始化融合引擎 =================
engine = FusionEngine();

% 注册探测器
engine.addDetector('RedEdge',      RedEdgeDetector());
engine.addDetector('Intrinsic',    IntrinsicDetector());
engine.addDetector('SlowVars',     SlowVarsDetector());

% 只有当使用了 KML 时才注册 KnownAnomaly 探测器
if any(strcmp(detectors_to_use, 'KnownAnomaly'))
    engine.addDetector('KnownAnomaly', KnownAnomalyDetector());
end

% ================= 6. 执行计算 =================
fprintf('>>> 开始并行计算各异常层...\n');
engine.computeAll(dataCtx);

% ================= 7. 结果融合 =================
% 使用我们定义的 detectors_to_use 列表进行融合
% 这样文件夹名字和实际用到的探测器就完全对应了
final_mask = engine.getFusedMask(detectors_to_use);

% ================= 8. 后处理与绘图 =================
PostProcessor.run(dataCtx, engine, final_mask, config.outDir);

fprintf('✅ 所有流程完成！结果路径：%s\n', config.outDir);