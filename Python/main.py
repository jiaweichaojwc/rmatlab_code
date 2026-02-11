"""
Main script for Schumann Resonance Remote Sensing System
Converted from MATLAB Main.m

面向对象主程序 - 全交互式读取 + 4掩码融合(含KML) + 动态文件夹命名
"""

import numpy as np
import sys
import os
from datetime import datetime
from tkinter import Tk, messagebox

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Core.geo_data_context import GeoDataContext
from Core.fusion_engine import FusionEngine
from Core.post_processor import PostProcessor
from Detectors.red_edge_detector import RedEdgeDetector
from Detectors.intrinsic_detector import IntrinsicDetector
from Detectors.slow_vars_detector import SlowVarsDetector
from Detectors.known_anomaly_detector import KnownAnomalyDetector


def main():
    """Main execution function"""
    print('='*60)
    print('舒曼波共振遥感 - 面向对象主程序')
    print('='*60)
    
    # ================= 1. Basic Configuration =================
    config = {
        'mineral_type': 'gold',
        'region_type': '',  # Empty triggers interactive mode
        'levashov_mode': True,
        'fusion_mode': True
    }
    
    # ================= 2. KML Known Anomaly Configuration (Interactive) =================
    print('>>> [交互模式] 是否导入 KML/KMZ 已知异常文件 (第4掩码)？')
    
    root = Tk()
    root.withdraw()
    answer = messagebox.askyesno('KML配置', '是否导入 KML/KMZ 已知异常文件？')
    root.destroy()
    
    # Define base detector list
    detectors_to_use = ['RedEdge', 'Intrinsic']
    
    if answer:
        from tkinter import filedialog
        root = Tk()
        root.withdraw()
        kml_file = filedialog.askopenfilename(
            title='请选择已知异常文件',
            filetypes=[('Google Earth Files', '*.kml *.kmz'), ('All files', '*.*')]
        )
        root.destroy()
        
        if kml_file:
            config['kmz_path'] = kml_file
            print(f'✅ 已选择 KML 文件: {kml_file}')
            
            # If KML is selected, add KnownAnomaly to list
            detectors_to_use.append('KnownAnomaly')
        else:
            config['kmz_path'] = ''
            print('⚠️ 用户取消选择 KML 文件，将跳过此步骤。')
    else:
        config['kmz_path'] = ''
        print('>>> 跳过 KML 导入。')
    
    # KML extraction keywords
    config['kmz_keywords'] = ['矿体投影', 'Object ID', 'ZK', '异常', '已知矿点']
    
    # ================= 3. Initialize Data Context =================
    # This will trigger GeoUtils.getRegionConfig to show folder selection dialog
    print('\n>>> 初始化数据上下文...')
    dataCtx = GeoDataContext(config)
    
    # ================= 4. Set Output Path (Dynamic Naming) =================
    # Concatenate detector types into string (e.g., RedEdge_Intrinsic_KnownAnomaly)
    types_str = '_'.join(detectors_to_use)
    
    # Construct folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    folder_name = f'{types_str}_Result_{config["mineral_type"]}_{timestamp}'
    config['outDir'] = os.path.join(dataCtx.data_dir, folder_name)
    
    if not os.path.exists(config['outDir']):
        os.makedirs(config['outDir'])
    print(f'📂 结果将保存至: {config["outDir"]}')
    
    # ================= 5. Initialize Fusion Engine =================
    print('\n>>> 初始化融合引擎...')
    engine = FusionEngine()
    
    # Register detectors
    engine.addDetector('RedEdge', RedEdgeDetector())
    engine.addDetector('Intrinsic', IntrinsicDetector())
    engine.addDetector('SlowVars', SlowVarsDetector())
    
    # Only register KnownAnomaly detector when KML is used
    if 'KnownAnomaly' in detectors_to_use:
        engine.addDetector('KnownAnomaly', KnownAnomalyDetector())
    
    # ================= 6. Execute Calculations =================
    print('\n>>> 开始并行计算各异常层...')
    engine.computeAll(dataCtx)
    
    # ================= 7. Result Fusion =================
    print('\n>>> 进行结果融合...')
    # Use our defined detectors_to_use list for fusion
    # This ensures folder name matches actually used detectors
    final_mask = engine.getFusedMask(detectors_to_use)
    
    # ================= 8. Post-processing and Plotting =================
    print('\n>>> 后处理与绘图...')
    PostProcessor.run(dataCtx, engine, final_mask, config['outDir'])
    
    print(f'\n✅ 所有流程完成！结果路径：{config["outDir"]}')
    print('='*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n❌ 用户中断程序')
        sys.exit(1)
    except Exception as e:
        print(f'\n\n❌ 程序出错: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
