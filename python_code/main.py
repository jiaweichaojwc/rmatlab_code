#!/usr/bin/env python3
"""
Schumann Resonance Remote Sensing - Main Program
舒曼波共振遥感 - Python主程序

This is the Python translation of Main.m
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core import GeoDataContext, FusionEngine, PostProcessor
from detectors import (
    RedEdgeDetector,
    IntrinsicDetector,
    KnownAnomalyDetector,
    SlowVarsDetector
)


def main():
    """Main processing pipeline."""
    
    print("=" * 80)
    print("舒曼波共振遥感 - 面向对象主程序 (Python版)")
    print("=" * 80)
    print()
    
    # ================= 1. Basic Configuration =================
    config = {
        'mineral_type': 'gold',
        'region_type': '',  # Empty for interactive mode (not supported in Python)
        'levashov_mode': True,
        'fusion_mode': True,
    }
    
    # ================= 2. Data and ROI Configuration =================
    # Note: In Python, we need to specify paths directly
    # Users should modify these paths according to their data location
    
    # Example paths (modify according to your setup):
    config['data_dir'] = './data'  # Path to data directory
    config['roi_file'] = './coordinates.xlsx'  # Path to ROI coordinate file
    
    # Check if paths exist
    if not os.path.exists(config['data_dir']):
        print("❌ 错误: 数据目录不存在，请在代码中设置正确的 data_dir 路径")
        print(f"   当前设置: {config['data_dir']}")
        print()
        print("提示: 编辑 main.py 文件，修改 config['data_dir'] 和 config['roi_file']")
        return 1
    
    # ================= 3. KML Known Anomaly Configuration =================
    print(">>> [配置] KML/KMZ 已知异常文件配置")
    
    # Define base detectors
    detectors_to_use = ['RedEdge', 'Intrinsic']
    
    # Ask about KML file (simplified - in practice, use command-line args or config file)
    use_kml = input("是否导入 KML/KMZ 已知异常文件? (y/n, 默认n): ").strip().lower()
    
    if use_kml in ['y', 'yes', '是']:
        kml_path = input("请输入 KML/KMZ 文件路径: ").strip()
        if kml_path and os.path.exists(kml_path):
            config['kmz_path'] = kml_path
            print(f"✅ 已选择 KML 文件: {kml_path}")
            detectors_to_use.append('KnownAnomaly')
        else:
            config['kmz_path'] = ''
            print("⚠️ 文件不存在，将跳过此步骤。")
    else:
        config['kmz_path'] = ''
        print(">>> 跳过 KML 导入。")
    
    # KML extraction keywords
    config['kmz_keywords'] = ['矿体投影', 'Object ID', 'ZK', '异常', '已知矿点']
    
    # ================= 4. Initialize Data Context =================
    print()
    print(">>> 初始化数据上下文...")
    
    try:
        data_ctx = GeoDataContext(config)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("提示: 请检查数据目录和ROI文件路径是否正确")
        return 1
    
    # ================= 5. Set Output Path =================
    # Create dynamic folder name with detector types and timestamp
    types_str = '_'.join(detectors_to_use)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    folder_name = f"{types_str}_Result_{config['mineral_type']}_{timestamp}"
    
    if data_ctx.data_dir:
        config['outDir'] = os.path.join(data_ctx.data_dir, folder_name)
    else:
        config['outDir'] = os.path.join('.', folder_name)
    
    os.makedirs(config['outDir'], exist_ok=True)
    print(f"📂 结果将保存至: {config['outDir']}")
    
    # ================= 6. Initialize Fusion Engine =================
    print()
    print(">>> 初始化融合引擎...")
    engine = FusionEngine()
    
    # Register detectors
    engine.add_detector('RedEdge', RedEdgeDetector())
    engine.add_detector('Intrinsic', IntrinsicDetector())
    engine.add_detector('SlowVars', SlowVarsDetector())
    
    # Only register KnownAnomaly if KML is used
    if 'KnownAnomaly' in detectors_to_use:
        engine.add_detector('KnownAnomaly', KnownAnomalyDetector())
    
    # ================= 7. Execute Computation =================
    print()
    print(">>> 开始并行计算各异常层...")
    
    try:
        engine.compute_all(data_ctx)
    except Exception as e:
        print(f"❌ 计算过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ================= 8. Result Fusion =================
    print()
    print(">>> 融合探测结果...")
    
    try:
        final_mask = engine.get_fused_mask(detectors_to_use)
    except Exception as e:
        print(f"❌ 融合过程出错: {e}")
        return 1
    
    # ================= 9. Post-processing and Visualization =================
    print()
    print(">>> 后处理与绘图...")
    
    try:
        PostProcessor.run(data_ctx, engine, final_mask, config['outDir'])
    except Exception as e:
        print(f"⚠️ 后处理警告: {e}")
    
    # ================= 10. Complete =================
    print()
    print("=" * 80)
    print(f"✅ 所有流程完成！结果路径：{config['outDir']}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
