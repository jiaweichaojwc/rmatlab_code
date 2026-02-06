%% ================= 舒曼波共振遥感 - 面向对象主程序 =================
% Main.m
clear all; close all; clc;
addpath(genpath(pwd)); % 确保加载所有类文件

% 1. 配置参数
config.mineral_type = 'gold'; 
config.region_type  = 'shanxi';
config.levashov_mode = true;
config.outDir       = ['Result_', config.mineral_type, '_', datestr(now, 'yyyymmdd')];
if ~exist(config.outDir, 'dir'), mkdir(config.outDir); end

% 2. 初始化数据上下文 (自动加载数据)
fprintf('>>> 初始化数据上下文...\n');
dataCtx = GeoDataContext(config);

% 3. 初始化融合引擎
engine = FusionEngine();

% 4. 注册探测器 (在这里解耦！你可以注释掉任意一行，或者添加新的)
%    每个探测器内部逻辑与原脚本完全一致
engine.addDetector('RedEdge',   RedEdgeDetector());        % 红边
engine.addDetector('Intrinsic', IntrinsicDetector());      % 本征吸收
engine.addDetector('SlowVars',  SlowVarsDetector());       % 七个慢变量

% 5. 执行计算
fprintf('>>> 开始并行计算各异常层...\n');
engine.computeAll(dataCtx);

% 6. 灵活融合 (核心需求：你可以随意组合)
% 方式A: 全融合 (原逻辑)
final_mask = engine.getFusedMask({'RedEdge', 'Intrinsic'});

% 方式B: 仅看两两结合 (示例)
%mask_two = engine.getFusedMask({'RedEdge', 'Intrinsic'});

% 7. 后续处理 (共振深度、地质约束、可视化)
%    注：为了保持简洁，后处理逻辑封装在 PostProcessor 中
PostProcessor.run(dataCtx, engine, final_mask, config.outDir);

fprintf('✅ 所有流程完成！\n');