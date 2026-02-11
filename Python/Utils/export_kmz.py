"""
exportKMZ - Export results to KMZ format
Converted from MATLAB exportKMZ.m
"""

import os
import sys
import subprocess


def exportKMZ(dataFile, outDir):
    """
    Export results to KMZ format by calling Python script
    
    Parameters:
    -----------
    dataFile : str
        Full path to .npz file storing mineral prediction data
    outDir : str
        Output directory path
    
    Returns:
    --------
    status : int
        0 for success, non-zero for failure
    """
    print('>>> [KMZ导出] 正在调用 Python 生成 KMZ...')
    
    # Auto-locate Python script path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    py_script = os.path.join(current_file_dir, '..', 'chengjie_matlab_code.py')
    py_script = os.path.abspath(py_script)
    
    # Check if Python script exists
    if not os.path.exists(py_script):
        print(f'❌ 未找到 Python 脚本！')
        print(f'预期路径: {py_script}')
        print('请确保将 chengjie_matlab_code.py 放入代码目录的 Python 文件夹中。')
        return 1
    
    # Use current Python interpreter
    py_exe = sys.executable
    
    # Construct command
    command = [py_exe, py_script, dataFile, outDir]
    
    try:
        # Execute command
        result = subprocess.run(command, capture_output=True, text=True, 
                              encoding='utf-8', errors='replace')
        
        if result.returncode == 0:
            print('✅ KMZ 导出成功！')
            print(f'📂 导出位置：{outDir}')
            return 0
        else:
            print('❌ KMZ 生成失败！')
            print('错误日志报告：')
            print('--------------------')
            print(result.stderr)
            print('--------------------')
            print('检查建议：')
            print(f'  1. 确认 Python 解释器路径: {py_exe}')
            print(f'  2. 确认 Python 脚本路径: {py_script}')
            print('  3. 确认已安装库: pip install simplekml pyproj scipy numpy matplotlib')
            return result.returncode
    
    except Exception as e:
        print(f'❌ 调用 Python 脚本时出错: {str(e)}')
        return 1
