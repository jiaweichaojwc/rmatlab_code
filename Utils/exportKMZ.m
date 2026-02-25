function status = exportKMZ(dataFile, outDir)
% exportKMZ - 调用 Python 脚本生成 KMZ 文件
%
% 输入参数:
%   dataFile - 存储成矿预测数据的 .mat 文件全路径
%   outDir   - 结果输出的文件夹路径

    fprintf('>>> [KMZ导出] 正在调用 Python 生成 KMZ...\n');

    % ================= 1. 自动定位 Python 脚本路径 =================
    % 获取当前 m 文件 (exportKMZ.m) 所在的目录 (即 Utils 文件夹)
    currentFileDir = fileparts(mfilename('fullpath'));
    
    % 假设 Python 脚本位于 Utils 的兄弟文件夹 Python 中
    % 路径拼接: Utils/../Python/chengjie_matlab_code.py
    py_script_rel = fullfile(currentFileDir, '..', 'Python', 'chengjie_matlab_code.py');
    
    % 获取绝对路径 (防止相对路径在 system 命令中出错)
    try
        py_script = char(java.io.File(py_script_rel).getCanonicalPath());
    catch
        % 如果 Java 调用失败，回退到简单拼接
        py_script = py_script_rel;
    end
    
    % 检查 Python 脚本是否存在
    if ~exist(py_script, 'file')
        error('GeoUtils:PythonScriptNotFound', ...
            '未找到 Python 脚本！\n预期路径: %s\n请确保将 chengjie_matlab_code.py 放入代码目录的 Python 文件夹中。', py_script);
    end

    % ================= 2. 设置 Python 解释器路径 =================
    % ⚠️⚠️⚠️ 请根据你的实际 Conda 环境修改此处 ⚠️⚠️⚠️
    % 之前代码中的路径:
    py_exe = 'C:\Users\Administrator\.conda\envs\scts\python.exe'; 
    
    % 如果你在 Deep-Lei 用户下，建议核实一下路径，例如:
    % py_exe = 'C:\Users\Deep-Lei\.conda\envs\scts\python.exe'; 
    %py_exe = C:\Users\Administrator\.conda\envs\scts\python.exe

    % ================= 3. 构造并执行指令 =================
    % 使用双引号包裹所有路径，防止空格或特殊字符导致错误
    commandStr = sprintf('"%s" "%s" "%s" "%s"', py_exe, py_script, dataFile, outDir);

    % 强制指定编码为 UTF-8，防止中文路径乱码
    feature('DefaultCharacterSet', 'UTF-8');

    % 调用系统执行
    [status, cmdMsg] = system(commandStr);

    % ================= 4. 结果反馈 =================
    if status == 0
        fprintf('✅ KMZ 导出成功！\n');
        fprintf('📂 导出位置：%s\n', outDir);
    else
        fprintf('❌ KMZ 生成失败！\n');
        fprintf('错误日志报告：\n--------------------\n%s\n--------------------\n', cmdMsg);
        fprintf('检查建议：\n');
        fprintf('  1. 确认 Python 解释器路径是否正确: %s\n', py_exe);
        fprintf('  2. 确认 Python 脚本路径是否正确: %s\n', py_script);
        fprintf('  3. 确认 conda 环境中已安装库: pip install simplekml pyproj scipy numpy matplotlib\n');
    end
end