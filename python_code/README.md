# 舒曼波共振遥感 - Python版本

Schumann Resonance Remote Sensing - Python Implementation

## 简介 (Introduction)

这是MATLAB版本的Python实现，用于基于舒曼波共振的矿产资源遥感探测。

This is a Python implementation of the MATLAB code for mineral resource remote sensing detection based on Schumann resonance.

## 主要功能 (Main Features)

- 多种异常探测器 (Multiple anomaly detectors):
  - 红边探测器 (Red Edge Detector)
  - 本征吸收探测器 (Intrinsic Absorption Detector)
  - 慢变量探测器 (Slow Variables Detector)
  - 已知异常探测器 (Known Anomaly Detector - KML/KMZ)

- 数据融合引擎 (Data Fusion Engine)
- 结果可视化 (Result Visualization)
- KMZ导出 (KMZ Export for Google Earth)

## 系统要求 (Requirements)

- Python 3.8 or higher
- 详见 requirements.txt

## 安装 (Installation)

1. 克隆或下载本代码库 (Clone or download this repository)

2. 安装依赖包 (Install dependencies):
```bash
pip install -r requirements.txt
```

## 使用方法 (Usage)

### 基本用法 (Basic Usage)

1. 准备数据 (Prepare your data):
   - Sentinel-2 L2A数据 (Sentinel-2 L2A data)
   - Landsat 8数据 (可选) (Landsat 8 data, optional)
   - ASTER L2数据 (可选) (ASTER L2 data, optional)
   - DEM数据 (DEM data)
   - ROI坐标文件 (Excel/CSV) (ROI coordinate file)

2. 修改配置 (Modify configuration):
   编辑 `main.py` 文件中的配置部分:
   ```python
   config['data_dir'] = './data'  # 数据目录路径
   config['roi_file'] = './coordinates.xlsx'  # ROI坐标文件路径
   ```

3. 运行程序 (Run the program):
```bash
python main.py
```

或者 (or):
```bash
python3 main.py
```

### 高级用法 (Advanced Usage)

#### 程序化调用 (Programmatic Usage)

```python
from core import GeoDataContext, FusionEngine
from detectors import RedEdgeDetector, IntrinsicDetector

# 配置 (Configuration)
config = {
    'mineral_type': 'gold',
    'data_dir': './data',
    'roi_file': './coordinates.xlsx',
    'levashov_mode': True
}

# 初始化数据 (Initialize data)
data_ctx = GeoDataContext(config)

# 创建融合引擎 (Create fusion engine)
engine = FusionEngine()
engine.add_detector('RedEdge', RedEdgeDetector())
engine.add_detector('Intrinsic', IntrinsicDetector())

# 计算 (Compute)
engine.compute_all(data_ctx)

# 获取结果 (Get results)
final_mask = engine.get_fused_mask(['RedEdge', 'Intrinsic'])
```

## 数据格式 (Data Format)

### ROI坐标文件 (ROI Coordinate File)

支持Excel (.xlsx, .xls) 或 CSV格式，应包含经纬度列:
- 经度列 (Longitude column): 数值范围 60-160
- 纬度列 (Latitude column): 数值范围 0-60

示例 (Example):
```
名称,经度,纬度
点1,105.23,35.67
点2,105.45,35.89
...
```

### 数据目录结构 (Data Directory Structure)

```
data/
├── Sentinel-2_L2A/
│   ├── *B02*.jp2
│   ├── *B03*.jp2
│   └── ...
├── Landsat_8_L1/  (可选)
│   ├── *B2*.tif
│   └── ...
├── ASTER_L2/  (可选)
│   └── ...
└── DEM.tif
```

## 输出结果 (Output)

程序会在数据目录下创建结果文件夹，包含:
- `final_mask.npy` - 最终异常掩码 (Final anomaly mask)
- `results_visualization.png` - 可视化结果 (Visualization)
- `detection_statistics.txt` - 统计报告 (Statistics report)

## 与MATLAB版本的差异 (Differences from MATLAB Version)

1. **交互式选择**: Python版本不支持GUI交互式文件选择，需要在代码中直接指定路径
   
2. **简化实现**: 某些功能为简化版本，特别是:
   - KML/KMZ处理 (需要额外安装geopandas等库)
   - 部分可视化功能

3. **依赖库**: Python版本需要安装多个科学计算和地理信息处理库

## 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

**Q: 提示"数据目录不存在"**

A: 请在 `main.py` 中正确设置 `config['data_dir']` 路径

**Q: 提示rasterio或其他库未安装**

A: 运行 `pip install -r requirements.txt` 安装所有依赖

**Q: KML功能不工作**

A: KML功能需要额外的地理信息库支持，确保安装了 geopandas, fiona 等

## 许可证 (License)

请参考项目根目录的LICENSE文件

## 联系方式 (Contact)

如有问题，请提交Issue到项目仓库

## 更新日志 (Changelog)

### v1.0.0 (2026-02-10)
- 初始Python实现
- 完整的探测器模块
- 基础可视化功能
