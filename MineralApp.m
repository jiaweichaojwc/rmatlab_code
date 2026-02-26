classdef MineralApp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure             matlab.ui.Figure
        GridLayout           matlab.ui.container.GridLayout
        LeftPanel            matlab.ui.container.Panel
        RightPanel           matlab.ui.container.Panel
        
        % Inputs
        DirLabel             matlab.ui.control.Label
        DirEdit              matlab.ui.control.EditField
        DirBtn               matlab.ui.control.Button
        
        ROILabel             matlab.ui.control.Label
        ROIEdit              matlab.ui.control.EditField
        ROIBtn               matlab.ui.control.Button
        
        MineralLabel         matlab.ui.control.Label
        MineralDropDown      matlab.ui.control.DropDown
        
        KMZCheckBox          matlab.ui.control.CheckBox
        KMZEdit              matlab.ui.control.EditField
        KMZBtn               matlab.ui.control.Button
        
        % Detectors
        DetectorPanel        matlab.ui.container.Panel
        cbRedEdge            matlab.ui.control.CheckBox
        cbIntrinsic          matlab.ui.control.CheckBox
        cbSlowVars           matlab.ui.control.CheckBox
        cbKnown              matlab.ui.control.CheckBox
        
        % KMZ Threshold (New)
        KMZThresholdLabel    matlab.ui.control.Label
        KMZThresholdEdit     matlab.ui.control.NumericEditField
        
        % Task Name (New)
        TaskNameLabel        matlab.ui.control.Label
        TaskNameEdit         matlab.ui.control.EditField
        
        % Actions
        RunBtn               matlab.ui.control.Button
        SaveAsBtn            matlab.ui.control.Button  % New
        RenameBtn            matlab.ui.control.Button  % New
        
        StatusArea           matlab.ui.control.TextArea
        
        % Visualization Tabs & Images
        ResultTabs           matlab.ui.container.TabGroup
        
        TabLog               matlab.ui.container.Tab
        
        TabResonance         matlab.ui.container.Tab
        ImageResonance       matlab.ui.control.Image
        
        TabFusion            matlab.ui.container.Tab
        ImageFusion          matlab.ui.control.Image
        
        TabPrediction        matlab.ui.container.Tab
        ImagePrediction      matlab.ui.control.Image
    end

    properties (Access = private)
        % Internal Data
        Config struct
        CurrentResultDir char = '' % 存储当前结果文件夹路径，供另存为/重命名使用
    end

    methods (Access = private)
        
        % Helper: Log message
        function log(app, msg)
            timestamp = datestr(now, 'HH:MM:SS');
            newMsg = sprintf('[%s] %s', timestamp, msg);
            app.StatusArea.Value = [app.StatusArea.Value; {newMsg}];
            fprintf('%s\n', newMsg); % Also print to command window
            scroll(app.StatusArea, 'bottom');
            drawnow;
        end

        % Callback: Select Data Directory
        function selectDataDir(app, ~)
            sel = uigetdir(pwd, '选择 Data 数据文件夹');
            if sel ~= 0
                app.DirEdit.Value = sel;
                app.log(sprintf('已选择数据目录: %s', sel));
            end
        end

        % Callback: Select ROI File
        function selectROI(app, ~)
            [f, p] = uigetfile({'*.xlsx';'*.xls';'*.csv'}, '选择坐标文件');
            if f ~= 0
                fullPath = fullfile(p, f);
                app.ROIEdit.Value = fullPath;
                app.log(sprintf('已选择坐标文件: %s', f));
            end
        end

        % Callback: Select KML File
        function selectKML(app, ~)
            [f, p] = uigetfile({'*.kml';'*.kmz'}, '选择已知异常 KML/KMZ');
            if f ~= 0
                fullPath = fullfile(p, f);
                app.KMZEdit.Value = fullPath;
                app.KMZCheckBox.Value = true;
                app.cbKnown.Value = true; % Auto check the detector
                app.log(sprintf('已选择 KML 文件: %s', f));
            end
        end

        % Callback: Save Results As... (New)
        function saveResultsAs(app, ~)
            if isempty(app.CurrentResultDir) || ~exist(app.CurrentResultDir, 'dir')
                uialert(app.UIFigure, '当前没有可保存的结果，请先运行分析。', '提示');
                return;
            end
            
            % 选择目标文件夹
            destFolder = uigetdir(pwd, '选择另存为的目标文件夹');
            if destFolder == 0, return; end
            
            [~, currentName] = fileparts(app.CurrentResultDir);
            newPath = fullfile(destFolder, currentName);
            
            try
                app.log(sprintf('正在复制结果到: %s ...', newPath));
                copyfile(app.CurrentResultDir, newPath,'f');
                app.log('✅ 另存为成功！');
                uialert(app.UIFigure, ['结果已成功保存到：' newPath], '成功');
            catch ME
                app.log(['❌ 另存为失败: ', ME.message]);
                uialert(app.UIFigure, ME.message, '保存失败');
            end
        end

        % Callback: Rename Result Folder (New)
        function renameResult(app, ~)
            if isempty(app.CurrentResultDir) || ~exist(app.CurrentResultDir, 'dir')
                uialert(app.UIFigure, '当前没有可重命名的结果，请先运行分析。', '提示');
                return;
            end
            
            [parentDir, currentName] = fileparts(app.CurrentResultDir);
            
            % 弹出输入框
            prompt = {'请输入新的结果文件夹名称:'};
            dlgtitle = '重命名结果';
            dims = [1 50];
            definput = {currentName};
            answer = inputdlg(prompt, dlgtitle, dims, definput);
            
            if isempty(answer), return; end
            newName = answer{1};
            
            % 检查非法字符
            if regexp(newName, '[<>:"/\\|?*]')
                uialert(app.UIFigure, '名称包含非法字符，请重试。', '错误');
                return;
            end
            
            newPath = fullfile(parentDir, newName);
            
            if strcmp(newPath, app.CurrentResultDir)
                return; % 名称没变
            end
            
            if exist(newPath, 'dir')
                uialert(app.UIFigure, '该名称的文件夹已存在，请换一个名字。', '命名冲突');
                return;
            end
            
            try
                movefile(app.CurrentResultDir, newPath);
                app.CurrentResultDir = newPath; % 更新内部记录
                app.log(sprintf('✅ 结果文件夹已重命名为: %s', newName));
                uialert(app.UIFigure, '重命名成功！', '成功');
            catch ME
                app.log(['❌ 重命名失败: ', ME.message]);
                uialert(app.UIFigure, ME.message, '失败');
            end
        end

        % Callback: Run Analysis
        function runAnalysis(app, ~)
            % 1. Validation
            if isempty(app.DirEdit.Value) || isempty(app.ROIEdit.Value)
                uialert(app.UIFigure, '请先选择数据文件夹和坐标文件！', '配置错误');
                return;
            end
            
            app.RunBtn.Enable = 'off';
            app.SaveAsBtn.Enable = 'off';
            app.RenameBtn.Enable = 'off';
            app.RunBtn.Text = '正在运行...';
            
            app.log('=== 开始新的分析任务 ===');
            
            % 清空旧图片
            app.ImageResonance.ImageSource = '';
            app.ImageFusion.ImageSource = '';
            app.ImagePrediction.ImageSource = '';
            app.ResultTabs.SelectedTab = app.TabLog; 
            app.CurrentResultDir = ''; % 重置当前结果路径
            
            try
                % 2. Build Config
                cfg = struct();
                cfg.mineral_type = app.MineralDropDown.Value;
                cfg.region_type = ''; 
                cfg.levashov_mode = true;
                cfg.fusion_mode = true;
                
                % [新增] 读取界面配置的 KMZ 阈值
                cfg.kmz_threshold = app.KMZThresholdEdit.Value;
                
                cfg.data_dir = app.DirEdit.Value;
                cfg.roi_file = app.ROIEdit.Value;
                
                % KML Config
                detectors_to_use = {};
                if app.cbRedEdge.Value, detectors_to_use{end+1} = 'RedEdge'; end
                if app.cbIntrinsic.Value, detectors_to_use{end+1} = 'Intrinsic'; end
                if app.cbSlowVars.Value, detectors_to_use{end+1} = 'SlowVars'; end
                
                if app.KMZCheckBox.Value && ~isempty(app.KMZEdit.Value)
                    cfg.kmz_path = app.KMZEdit.Value;
                    cfg.kmz_keywords = {'矿体投影', 'Object ID', 'ZK', '异常', '已知矿点'};
                    if app.cbKnown.Value
                        detectors_to_use{end+1} = 'KnownAnomaly';
                    end
                    app.log('KML 模块已启用');
                else
                    cfg.kmz_path = '';
                end
                
                if isempty(detectors_to_use)
                    error('请至少选择一个探测器！');
                end
                
                % 3. Initialize Context
                app.log('正在初始化数据上下文 (GeoDataContext)...');
                dataCtx = GeoDataContext(cfg);
                
                % 4. Output Path Construction (支持自定义任务名)
                if ~isempty(app.TaskNameEdit.Value)
                    % 使用自定义名称
                    folder_name = app.TaskNameEdit.Value;
                    app.log(['使用自定义任务名称: ', folder_name]);
                else
                    % 使用默认时间戳名称
                    types_str = strjoin(detectors_to_use, '_');
                    folder_name = [types_str, '_Result_', cfg.mineral_type, '_', datestr(now, 'yyyymmdd_HHMM')];
                end
                
                cfg.outDir = fullfile(dataCtx.data_dir, folder_name);
                if ~exist(cfg.outDir, 'dir'), mkdir(cfg.outDir); end
                
                app.CurrentResultDir = cfg.outDir; % 记录下来供后续使用
                app.log(sprintf('结果输出路径: %s', cfg.outDir));
                
                % 5. Fusion Engine
                app.log('初始化融合引擎...');
                engine = FusionEngine();
                
                if any(strcmp(detectors_to_use, 'RedEdge')), engine.addDetector('RedEdge', RedEdgeDetector()); end
                if any(strcmp(detectors_to_use, 'Intrinsic')), engine.addDetector('Intrinsic', IntrinsicDetector()); end
                if any(strcmp(detectors_to_use, 'SlowVars')), engine.addDetector('SlowVars', SlowVarsDetector()); end
                if any(strcmp(detectors_to_use, 'KnownAnomaly')), engine.addDetector('KnownAnomaly', KnownAnomalyDetector()); end
                
                % 6. Compute
                app.log('开始计算各异常层 (ComputeAll)...');
                engine.computeAll(dataCtx);
                
                % 7. Fusion & PostProcess
                app.log('执行结果融合与后处理...');
                final_mask = engine.getFusedMask(detectors_to_use);
                PostProcessor.run(dataCtx, engine, final_mask, cfg.outDir);
                
                app.log('✅ 所有流程完成！');
                app.log(['结果已保存至: ', cfg.outDir]);
                
                % 8. Load Images
                % (1) Resonance
                img01 = fullfile(cfg.outDir, '01_共振参数综合图.png');
                if exist(img01, 'file'), app.ImageResonance.ImageSource = img01; end
                
                % (2) Fusion
                dir02 = dir(fullfile(cfg.outDir, '02_掩码集成_*.png'));
                if ~isempty(dir02)
                    img02 = fullfile(dir02(1).folder, dir02(1).name);
                    app.ImageFusion.ImageSource = img02;
                end
                
                % (3) Prediction
                img03 = fullfile(cfg.outDir, '03_深部成矿预测图.png');
                if exist(img03, 'file')
                    app.ImagePrediction.ImageSource = img03;
                    app.ResultTabs.SelectedTab = app.TabPrediction;
                end
                
                % 启用后续操作按钮
                app.SaveAsBtn.Enable = 'on';
                app.RenameBtn.Enable = 'on';
                
            catch ME
                app.log(['❌ 错误: ', ME.message]);
                disp(getReport(ME)); 
                uialert(app.UIFigure, ME.message, '运行错误');
            end
            
            app.RunBtn.Enable = 'on';
            app.RunBtn.Text = '开始运行分析';
        end
    end

    % App initialization and layout
    methods (Access = public)
        function createComponents(app)
            % Main Figure
            app.UIFigure = uifigure('Position', [100, 100, 1150, 720]); 
            app.UIFigure.Name = '舒曼波共振遥感 - 智能分析系统';
            
            % Grid Layout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {340, '1x'}; % 左侧面板加宽适应新按钮
            app.GridLayout.RowHeight = {'1x'};
            
            % --- Left Panel (Controls) ---
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;
            app.LeftPanel.Title = '参数配置';
            app.LeftPanel.FontSize = 14; 
            app.LeftPanel.FontWeight = 'bold';
            
            lpLayout = uigridlayout(app.LeftPanel);
            lpLayout.ColumnWidth = {'1x', 40};
            
            % [修改] 增加行高，容纳置信度控件
            lpLayout.RowHeight = {22, 25, 22, 25, 22, 25, 22, 25, 220, 22, 25, 22, 25, 50, 40, '1x'};
            
            % 1. Data Dir
            app.DirLabel = uilabel(lpLayout, 'Text', '1. Data 数据文件夹:');
            app.DirLabel.FontSize = 13;
            app.DirLabel.Layout.Row = 1; app.DirLabel.Layout.Column = [1 2];
            
            app.DirEdit = uieditfield(lpLayout, 'Editable', false);
            app.DirEdit.Layout.Row = 2; app.DirEdit.Layout.Column = 1;
            
            app.DirBtn = uibutton(lpLayout, 'Text', '...');
            app.DirBtn.Layout.Row = 2; app.DirBtn.Layout.Column = 2;
            app.DirBtn.ButtonPushedFcn = createCallbackFcn(app, @selectDataDir, true);
            
            % 2. ROI File
            app.ROILabel = uilabel(lpLayout, 'Text', '2. 坐标文件 (.xlsx):');
            app.ROILabel.FontSize = 13;
            app.ROILabel.Layout.Row = 3; app.ROILabel.Layout.Column = [1 2];
            
            app.ROIEdit = uieditfield(lpLayout, 'Editable', false);
            app.ROIEdit.Layout.Row = 4; app.ROIEdit.Layout.Column = 1;
            
            app.ROIBtn = uibutton(lpLayout, 'Text', '...');
            app.ROIBtn.Layout.Row = 4; app.ROIBtn.Layout.Column = 2;
            app.ROIBtn.ButtonPushedFcn = createCallbackFcn(app, @selectROI, true);
            
            % 3. Mineral Type
            app.MineralLabel = uilabel(lpLayout, 'Text', '3. 目标矿种:');
            app.MineralLabel.FontSize = 13;
            app.MineralLabel.Layout.Row = 5; app.MineralLabel.Layout.Column = [1 2];
            
            app.MineralDropDown = uidropdown(lpLayout);
            app.MineralDropDown.Items = {'gold', 'copper','cave', 'iron', 'lead', 'zinc', 'coal', 'petroleum', 'gas', 'rare_earth', 'lithium'};
            app.MineralDropDown.Value = 'gold';
            app.MineralDropDown.FontSize = 13;
            app.MineralDropDown.Layout.Row = 6; app.MineralDropDown.Layout.Column = [1 2];
            
            % 4. KML Config
            app.KMZCheckBox = uicheckbox(lpLayout, 'Text', '导入 KML/KMZ 已知异常');
            app.KMZCheckBox.FontSize = 13;
            app.KMZCheckBox.Layout.Row = 7; app.KMZCheckBox.Layout.Column = [1 2];
            
            app.KMZEdit = uieditfield(lpLayout, 'Editable', false);
            app.KMZEdit.Placeholder = 'KML 文件路径...';
            app.KMZEdit.Layout.Row = 8; app.KMZEdit.Layout.Column = 1;
            
            app.KMZBtn = uibutton(lpLayout, 'Text', '...');
            app.KMZBtn.Layout.Row = 8; app.KMZBtn.Layout.Column = 2;
            app.KMZBtn.ButtonPushedFcn = createCallbackFcn(app, @selectKML, true);
            
            % 5. Detectors Panel
            app.DetectorPanel = uipanel(lpLayout, 'Title', '启用的探测器 (多选)');
            app.DetectorPanel.Layout.Row = 9; app.DetectorPanel.Layout.Column = [1 2];
            app.DetectorPanel.FontSize = 14; 
            app.DetectorPanel.FontWeight = 'bold';
            
            dpLayout = uigridlayout(app.DetectorPanel, 'ColumnWidth', {'1x'}, 'RowHeight', {'1x','1x','1x','1x'});
            dpLayout.RowSpacing = 8;
            dpLayout.Padding = [20 10 20 10];
            
            app.cbRedEdge = uicheckbox(dpLayout, 'Text', 'RedEdge (红边)', 'Value', true, 'FontSize', 14, 'FontWeight', 'bold');
            app.cbRedEdge.Layout.Row = 1;
            app.cbIntrinsic = uicheckbox(dpLayout, 'Text', 'Intrinsic (本征吸收)', 'Value', true, 'FontSize', 14, 'FontWeight', 'bold');
            app.cbIntrinsic.Layout.Row = 2;
            app.cbSlowVars = uicheckbox(dpLayout, 'Text', 'SlowVars (慢变量)', 'Value', false, 'FontSize', 14, 'FontWeight', 'bold');
            app.cbSlowVars.Layout.Row = 3;
            app.cbKnown = uicheckbox(dpLayout, 'Text', 'KnownAnomaly (KML)', 'Value', false, 'FontSize', 14, 'FontWeight', 'bold');
            app.cbKnown.Layout.Row = 4;
            
            % 6. KMZ 导出置信度阈值 (New)
            app.KMZThresholdLabel = uilabel(lpLayout, 'Text', '4. 生成KMZ置信度 (0~1):');
            app.KMZThresholdLabel.FontSize = 13;
            app.KMZThresholdLabel.Layout.Row = 10; app.KMZThresholdLabel.Layout.Column = [1 2];
            
            app.KMZThresholdEdit = uieditfield(lpLayout, 'numeric');
            app.KMZThresholdEdit.Limits = [0.1 1.0];
            app.KMZThresholdEdit.Value = 0.6; % 默认值0.6
            app.KMZThresholdEdit.Layout.Row = 11; app.KMZThresholdEdit.Layout.Column = [1 2];
            
            % 7. Task Name
            app.TaskNameLabel = uilabel(lpLayout, 'Text', '5. 任务名称 (可选，留空则自动命名):');
            app.TaskNameLabel.FontSize = 13;
            app.TaskNameLabel.FontColor = [0.4 0.4 0.4]; % 灰色提示
            app.TaskNameLabel.Layout.Row = 12; app.TaskNameLabel.Layout.Column = [1 2];
            
            app.TaskNameEdit = uieditfield(lpLayout);
            app.TaskNameEdit.Placeholder = '例如: 新疆金矿_测试01';
            app.TaskNameEdit.Layout.Row = 13; app.TaskNameEdit.Layout.Column = [1 2];
            
            % 8. Run Button
            app.RunBtn = uibutton(lpLayout, 'Text', '开始运行分析', ...
                'BackgroundColor', [0.2, 0.6, 0.2], 'FontColor', 'w', 'FontWeight', 'bold', 'FontSize', 16);
            app.RunBtn.Layout.Row = 14; app.RunBtn.Layout.Column = [1 2];
            app.RunBtn.ButtonPushedFcn = createCallbackFcn(app, @runAnalysis, true);
            
            % 9. Action Buttons (Grid inside Grid for side-by-side buttons)
            actionGrid = uigridlayout(lpLayout, 'RowHeight', {'1x'}, 'ColumnWidth', {'1x', '1x'});
            actionGrid.Layout.Row = 15; actionGrid.Layout.Column = [1 2];
            actionGrid.Padding = [0 0 0 0];
            
            app.SaveAsBtn = uibutton(actionGrid, 'Text', '另存结果为...', 'Enable', 'off');
            app.SaveAsBtn.ButtonPushedFcn = createCallbackFcn(app, @saveResultsAs, true);
            
            app.RenameBtn = uibutton(actionGrid, 'Text', '重命名结果', 'Enable', 'off');
            app.RenameBtn.ButtonPushedFcn = createCallbackFcn(app, @renameResult, true);
            
            % --- Right Panel (Visualization & Logs) ---
            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;
            app.RightPanel.BorderType = 'none';
            
            rpLayout = uigridlayout(app.RightPanel, 'RowHeight', {'1x'}, 'ColumnWidth', {'1x'});
            
            % Create Tab Group
            app.ResultTabs = uitabgroup(rpLayout);
            app.ResultTabs.Layout.Row = 1; app.ResultTabs.Layout.Column = 1;
            
            % Tab 1: 运行日志
            app.TabLog = uitab(app.ResultTabs, 'Title', '运行日志');
            logLayout = uigridlayout(app.TabLog, 'ColumnWidth', {'1x'}, 'RowHeight', {'1x'});
            app.StatusArea = uitextarea(logLayout, 'Editable', false);
            app.StatusArea.FontSize = 12;
            app.StatusArea.Layout.Row = 1; app.StatusArea.Layout.Column = 1;
            app.StatusArea.Value = {'=== 系统就绪，请配置参数 ==='};
            
            % Tab 2: 共振参数
            app.TabResonance = uitab(app.ResultTabs, 'Title', '1. 共振参数');
            resLayout = uigridlayout(app.TabResonance, 'ColumnWidth', {'1x'}, 'RowHeight', {'1x'});
            app.ImageResonance = uiimage(resLayout);
            app.ImageResonance.ScaleMethod = 'fit';
            app.ImageResonance.Layout.Row = 1; app.ImageResonance.Layout.Column = 1;
            
            % Tab 3: 掩码集成
            app.TabFusion = uitab(app.ResultTabs, 'Title', '2. 掩码集成');
            fusLayout = uigridlayout(app.TabFusion, 'ColumnWidth', {'1x'}, 'RowHeight', {'1x'});
            app.ImageFusion = uiimage(fusLayout);
            app.ImageFusion.ScaleMethod = 'fit';
            app.ImageFusion.Layout.Row = 1; app.ImageFusion.Layout.Column = 1;
            
            % Tab 4: 深部预测
            app.TabPrediction = uitab(app.ResultTabs, 'Title', '3. 深部预测');
            predLayout = uigridlayout(app.TabPrediction, 'ColumnWidth', {'1x'}, 'RowHeight', {'1x'});
            app.ImagePrediction = uiimage(predLayout);
            app.ImagePrediction.ScaleMethod = 'fit';
            app.ImagePrediction.Layout.Row = 1; app.ImagePrediction.Layout.Column = 1;
            
        end

        function app = MineralApp()
            createComponents(app);
            registerApp(app, app.UIFigure);
        end
    end
end