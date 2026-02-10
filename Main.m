%% ================= 舒曼波共振遥感 - 面向对象主程序 =================
% Main.m
% 对应 untitled3.m 的逻辑：保留交互式读取，集成KML已知异常融合
clear all; close all; clc;
addpath(genpath(pwd)); 

% ================= 1. 基础配置 =================
config.mineral_type = 'gold'; 

% 【关键】置为空字符串，触发 GeoUtils 中的 uigetdir 交互式选择数据文件夹
config.region_type  = ''; 

config.levashov_mode = true;
config.fusion_mode = true;

% ================= 2. KML 已知异常配置 (交互式选择) =================
fprintf('>>> [交互模式] 是否导入 KML/KMZ 已知异常文件？\n');
answer = questdlg('是否导入 KML/KMZ 已知异常文件？', 'KML配置', '是', '否', '是');

if strcmp(answer, '是')
    [kml_file, kml_path] = uigetfile({'*.kml;*.kmz', 'Google Earth Files (*.kml, *.kmz)'}, '请选择已知异常文件');
    if kml_file ~= 0
        config.kmz_path = fullfile(kml_path, kml_file);
        fprintf('✅ 已选择 KML 文件: %s\n', config.kmz_path);
    else
        config.kmz_path = '';
        fprintf('⚠️ 用户取消选择 KML 文件，将跳过此步骤。\n');
    end
else
    config.kmz_path = '';
    fprintf('>>> 跳过 KML 导入。\n');
end

% 提取关键词配置 (与 untitled3.m 一致)
config.kmz_keywords = {'矿体投影', 'Object ID', 'ZK', '异常', '已知矿点'}; 

% ================= 3. 初始化数据上下文 =================
% 这里会触发 GeoUtils.getRegionConfig 弹窗选择数据文件夹
dataCtx = GeoDataContext(config);

% ================= 4. 设定输出路径 =================
folder_name = ['Result_', config.mineral_type, '_', datestr(now, 'yyyymmdd_HHMM')];
config.outDir = fullfile(dataCtx.data_dir, folder_name);

if ~exist(config.outDir, 'dir')
    mkdir(config.outDir);
end

% ================= 5. 初始化融合引擎 =================
engine = FusionEngine();

% 注册探测器 (加入 KnownAnomaly)
engine.addDetector('RedEdge',      RedEdgeDetector());
engine.addDetector('Intrinsic',    IntrinsicDetector());
engine.addDetector('SlowVars',     SlowVarsDetector());
engine.addDetector('KnownAnomaly', KnownAnomalyDetector()); % [新增：对应 Step 3.5]

% ================= 6. 执行计算 =================
fprintf('>>> 开始并行计算各异常层...\n');
engine.computeAll(dataCtx);

% ================= 7. 结果融合 =================
% [修改] 融合列表中加入 KnownAnomaly，对应 untitled3.m 的 mask1|mask2|mask3|mask4
final_mask = engine.getFusedMask({'RedEdge', 'Intrinsic', 'SlowVars', 'KnownAnomaly'});

% ================= 8. 后处理与绘图 =================
% 进入 PostProcessor 处理 Step 4 的增强融合与 Step 5 的可视化
PostProcessor.run(dataCtx, engine, final_mask, config.outDir);

fprintf('✅ 所有流程完成！结果路径：%s\n', config.outDir);