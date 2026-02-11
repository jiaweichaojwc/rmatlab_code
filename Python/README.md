# Schumann Resonance Remote Sensing System - Python Version

## 舒曼波共振遥感系统 - Python实现

This is a Python conversion of the MATLAB-based Schumann Resonance Remote Sensing system for mineral prediction.

本项目是基于MATLAB的舒曼波共振遥感系统的Python转换版本，用于矿产预测。

## Features / 功能特性

- **Red Edge Anomaly Detection** / 红边异常探测
- **Intrinsic Absorption Detection** / 本征吸收探测
- **Slow Variable Detection** / 慢变量探测
- **Known Anomaly Import (KML/KMZ)** / 已知异常导入(KML/KMZ)
- **Multi-layer Fusion** / 多层融合
- **Deep Prediction Visualization** / 深部预测可视化
- **KMZ Export** / KMZ导出

## Installation / 安装

### Prerequisites / 先决条件

- Python 3.7 or higher / Python 3.7或更高版本
- pip package manager / pip包管理器

### Install Dependencies / 安装依赖

```bash
cd Python
pip install -r requirements.txt
```

## Directory Structure / 目录结构

```
Python/
├── main.py                      # Main entry point / 主程序入口
├── requirements.txt             # Dependencies / 依赖列表
├── README.md                    # This file / 本文件
├── Core/                        # Core modules / 核心模块
│   ├── __init__.py
│   ├── geo_data_context.py     # Geographic data context / 地理数据上下文
│   ├── fusion_engine.py        # Detector fusion engine / 探测器融合引擎
│   └── post_processor.py       # Post-processing / 后处理
├── Detectors/                   # Anomaly detectors / 异常探测器
│   ├── __init__.py
│   ├── anomaly_detector.py     # Base class / 基类
│   ├── red_edge_detector.py    # Red edge detector / 红边探测器
│   ├── intrinsic_detector.py   # Intrinsic detector / 本征探测器
│   ├── slow_vars_detector.py   # Slow variables detector / 慢变量探测器
│   └── known_anomaly_detector.py # Known anomaly detector / 已知异常探测器
└── Utils/                       # Utility functions / 工具函数
    ├── __init__.py
    ├── geo_utils.py            # Geospatial utilities / 地理工具
    ├── visualizer.py           # Visualization / 可视化
    ├── kmz_mask_generator.py   # KMZ mask generator / KMZ掩码生成器
    └── export_kmz.py           # KMZ export / KMZ导出
```

## Usage / 使用方法

### Running the System / 运行系统

```bash
cd Python
python main.py
```

The system will:
1. Prompt you to select the data folder / 提示选择数据文件夹
2. Prompt you to select the coordinate file (Excel/CSV) / 提示选择坐标文件
3. Ask if you want to import KML/KMZ known anomalies / 询问是否导入KML/KMZ已知异常
4. Process all detectors and generate results / 处理所有探测器并生成结果

### Input Data Requirements / 输入数据要求

Your data folder should contain:
数据文件夹应包含：

```
data/
├── Sentinel 2 L2/              # Sentinel-2 L2A imagery / Sentinel-2 L2A影像
│   ├── B02.tif                # Blue band / 蓝波段
│   ├── B03.tif                # Green band / 绿波段
│   ├── B04.tif                # Red band / 红波段
│   ├── B05.tif                # Red Edge 1 / 红边1
│   ├── B06.tif                # Red Edge 2 / 红边2
│   ├── B07.tif                # Red Edge 3 / 红边3
│   ├── B08.tif                # NIR / 近红外
│   ├── B11.tif                # SWIR 1 / 短波红外1
│   └── B12.tif                # SWIR 2 / 短波红外2
├── Landsat 8 L1/ or L2/        # Landsat 8 imagery / Landsat 8影像
│   ├── B2.tif                 # Blue / 蓝
│   ├── B3.tif                 # Green / 绿
│   ├── B4.tif                 # Red / 红
│   ├── B5.tif                 # NIR / 近红外
│   ├── B6.tif                 # SWIR 1 / 短波红外1
│   └── B7.tif                 # SWIR 2 / 短波红外2
├── ASTER L1/ or L2/            # ASTER imagery / ASTER影像
│   ├── B01.tif - B09.tif      # VNIR and SWIR bands / 可见光近红外短波红外波段
│   └── B10.tif - B14.tif      # TIR bands / 热红外波段
└── DEM.tif                     # Digital Elevation Model / 数字高程模型
```

Coordinate file (Excel or CSV):
坐标文件（Excel或CSV）：

- Should contain longitude and latitude columns / 应包含经度和纬度列
- The system will auto-detect the columns / 系统会自动识别列

### Output / 输出

Results will be saved in a timestamped folder:
结果将保存在带时间戳的文件夹中：

```
[Detectors]_Result_[mineral_type]_[timestamp]/
├── 01_共振参数综合图.png           # Resonance parameters / 共振参数
├── 02_掩码集成_N图.png             # Fused masks / 融合掩码
├── 03_深部成矿预测图.png           # Deep prediction / 深部预测
├── 03_深部成矿预测图.pkl          # Editable figure / 可编辑图形
├── [mineral_type]_Result.npz      # Numerical results / 数值结果
└── [mineral_type]_Result.kmz      # Google Earth overlay / Google Earth叠加
```

## Key Differences from MATLAB / 与MATLAB版本的主要差异

### Array Indexing / 数组索引
- MATLAB: 1-indexed / 基于1的索引
- Python: 0-indexed / 基于0的索引
- **All band indices have been carefully adjusted** / 所有波段索引已仔细调整

### Data Structures / 数据结构
- MATLAB `containers.Map` → Python `dict`
- MATLAB cell arrays → Python lists
- MATLAB structs → Python dicts

### File I/O / 文件输入输出
- MATLAB `.mat` files → Python `.npz` files (numpy format)
- MATLAB `readgeoraster` → Python `rasterio`
- MATLAB `geotiffinfo` → Python `rasterio`

### Image Processing / 图像处理
- MATLAB `imresize` → Python `scipy.ndimage.zoom`
- MATLAB `imgaussfilt` → Python `scipy.ndimage.gaussian_filter`
- MATLAB `edge`, `bwareaopen` → Python `scikit-image`

### Dialogs / 对话框
- MATLAB `questdlg`, `uigetfile` → Python `tkinter` dialogs

## Mathematical Equivalence / 数学等价性

All mathematical formulas and algorithms are **EXACTLY** preserved:
所有数学公式和算法都被**精确**保留：

- Red Edge Position (S2REP) calculation / 红边位置计算
- Moran I spatial autocorrelation / Moran I空间自相关
- Intrinsic absorption features / 本征吸收特征
- Slow variable discriminant analysis / 慢变量判别分析
- Yakymchuk depth-frequency model / Yakymchuk深度频率模型

## Testing / 测试

To verify mathematical equivalence with MATLAB:
要验证与MATLAB的数学等价性：

1. Run both versions on the same dataset / 在同一数据集上运行两个版本
2. Compare numerical outputs in `.npz` (Python) vs `.mat` (MATLAB) / 比较`.npz`和`.mat`中的数值输出
3. Visual comparison of generated figures / 生成图形的视觉比较

## Troubleshooting / 故障排除

### Missing Dependencies / 缺少依赖
```bash
pip install --upgrade -r requirements.txt
```

### tkinter Not Found / 找不到tkinter
On Linux:
```bash
sudo apt-get install python3-tk
```

### Memory Issues / 内存问题
For large datasets, increase available memory or process tiles separately.
对于大数据集，增加可用内存或分块处理。

## License / 许可证

Same as the original MATLAB version.
与原MATLAB版本相同。

## Contact / 联系方式

For questions or issues, please contact the development team.
如有问题，请联系开发团队。

## Citation / 引用

If you use this system in your research, please cite:
如果您在研究中使用此系统，请引用：

[Add appropriate citation here]
[在此添加适当的引用]
