# Remote Sensing Anomaly Detection - Python Version
# 遥感异常探测 - Python版本

> **注意 / Note**: 此分支仅包含Python代码 / This branch contains **ONLY Python code**

## 简介 / Introduction

这是基于舒曼波共振的矿产资源遥感探测系统的Python实现。

This is a Python implementation of a mineral resource remote sensing detection system based on Schumann resonance.

## 快速开始 / Quick Start

### 安装依赖 / Install Dependencies

```bash
pip install -r python_code/requirements.txt
```

### 配置数据路径 / Configure Data Paths

编辑 `python_code/main.py` 文件，设置数据目录和ROI文件路径：

Edit `python_code/main.py` to set your data directory and ROI file path:

```python
config['data_dir'] = './data'              # 数据目录路径
config['roi_file'] = './coordinates.xlsx'  # ROI坐标文件路径
```

### 运行程序 / Run the Program

```bash
# 方法1: 直接运行 / Method 1: Direct run
python python_code/main.py

# 方法2: 使用帮助脚本 / Method 2: Use helper script
python run_detection.py
```

## 项目结构 / Project Structure

```
.
├── README.md                    # 本文件 / This file
├── python_code/                 # Python代码包 / Python package
│   ├── README.md               # 详细文档 / Detailed documentation
│   ├── requirements.txt        # Python依赖 / Dependencies
│   ├── main.py                 # 主程序 / Main program
│   ├── core/                   # 核心模块 / Core modules
│   │   ├── geo_data_context.py
│   │   ├── fusion_engine.py
│   │   └── post_processor.py
│   ├── detectors/              # 探测器 / Detectors
│   │   ├── anomaly_detector.py
│   │   ├── red_edge_detector.py
│   │   ├── intrinsic_detector.py
│   │   ├── known_anomaly_detector.py
│   │   └── slow_vars_detector.py
│   └── utils/                  # 工具类 / Utilities
│       ├── geo_utils.py
│       ├── kmz_mask_generator.py
│       ├── visualizer.py
│       └── export_kmz.py
├── run_detection.py            # 运行脚本 / Run script
└── PYTHON_BRANCH_SUMMARY.md    # 转换说明 / Conversion notes
```

## 主要功能 / Main Features

- ✅ **红边探测器** / Red Edge Detector
- ✅ **本征吸收探测器** / Intrinsic Absorption Detector
- ✅ **慢变量探测器** / Slow Variables Detector
- ✅ **已知异常探测器** / Known Anomaly Detector (KML/KMZ)
- ✅ **多探测器融合** / Multi-detector Fusion
- ✅ **结果可视化** / Result Visualization

## 数据要求 / Data Requirements

### 输入数据 / Input Data

- **Sentinel-2 L2A** 数据 / data
- **Landsat 8** 数据（可选）/ data (optional)
- **ASTER L2** 数据（可选）/ data (optional)
- **DEM** 数字高程模型 / Digital Elevation Model
- **ROI** 坐标文件 (Excel/CSV) / coordinate file

### ROI文件格式 / ROI File Format

Excel或CSV文件，包含经纬度列：

Excel or CSV file with longitude and latitude columns:

| 名称/Name | 经度/Longitude | 纬度/Latitude |
|-----------|----------------|---------------|
| 点1       | 105.23         | 35.67         |
| 点2       | 105.45         | 35.89         |

## 输出结果 / Output

程序会创建结果目录，包含：

The program creates a result directory containing:

- `final_mask.npy` - 最终异常掩码 / Final anomaly mask
- `results_visualization.png` - 可视化结果 / Visualization
- `detection_statistics.txt` - 统计报告 / Statistics report

## 技术栈 / Technology Stack

- **Python 3.8+**
- **NumPy** - 数值计算 / Numerical computing
- **SciPy** - 科学计算 / Scientific computing
- **scikit-image** - 图像处理 / Image processing
- **rasterio** - 地理空间数据 / Geospatial data
- **matplotlib** - 可视化 / Visualization

详细依赖请查看 `python_code/requirements.txt`

See `python_code/requirements.txt` for detailed dependencies

## 文档 / Documentation

- 📖 [完整使用文档 / Full Documentation](python_code/README.md)
- 📋 [转换说明 / Conversion Notes](PYTHON_BRANCH_SUMMARY.md)
- ✅ [完成状态 / Completion Status](CONVERSION_COMPLETE.txt)

## 原MATLAB版本 / Original MATLAB Version

原MATLAB代码位于主分支（main branch），本分支仅包含Python实现。

The original MATLAB code is on the main branch. This branch contains only the Python implementation.

## 许可证 / License

请参考项目根目录的LICENSE文件

Please refer to the LICENSE file in the project root

## 贡献 / Contributing

欢迎提交问题和改进建议！

Issues and improvements are welcome!

---

**开发者 / Developer**: Developer  
**日期 / Date**: 2026-02-10  
**分支 / Branch**: python (Python-only)
