%% ================= 舒曼波共振遥感 - 面向对象主程序 =================
% Main.m
clear all; close all; clc;
addpath(genpath(pwd)); 

% ================= 1. 基础配置 =================
config.mineral_type = 'gold'; 
config.region_type  = '';     % 空值触发交互选择
config.levashov_mode = true;

% ================= 2. 初始化数据上下文 =================
fprintf('>>> 初始化数据上下文 (请在弹窗中选择数据)...\n');
% 这一步会执行 GeoUtils.getRegionConfig -> 弹窗选择 -> 加载数据
dataCtx = GeoDataContext(config);

% ================= 3. 【核心修改】后置设定输出路径 =================
% 现在数据加载完了，我们可以从 dataCtx 里问出刚才选了哪个文件夹
folder_name = ['Result_', config.mineral_type, '_', datestr(now, 'yyyymmdd')];

% 直接拼接到 data_dir 里面
config.outDir = fullfile(dataCtx.data_dir, folder_name);

% 创建文件夹
if ~exist(config.outDir, 'dir')
    mkdir(config.outDir);
    fprintf('✅ 已创建结果目录：%s\n', config.outDir);
else
    fprintf('⚠️ 结果目录已存在：%s\n', config.outDir);
end

% ================= 4. 初始化融合引擎 =================
engine = FusionEngine();

% 注册探测器
engine.addDetector('RedEdge',   RedEdgeDetector());
engine.addDetector('Intrinsic', IntrinsicDetector());
engine.addDetector('SlowVars',  SlowVarsDetector());

% ================= 5. 执行计算 =================
fprintf('>>> 开始并行计算各异常层...\n');
engine.computeAll(dataCtx);

% ================= 6. 结果融合 =================
final_mask = engine.getFusedMask({'RedEdge', 'Intrinsic', 'SlowVars'});

% ================= 7. 后处理与绘图 =================
% 传入我们刚才生成的 config.outDir
PostProcessor.run(dataCtx, engine, final_mask, config.outDir);

fprintf('✅ 所有流程完成！结果路径：%s\n', config.outDir);