# Schumann Resonance Remote Sensing - Python Implementation

这是一个完全用 Python 实现的舒曼波共振遥感矿产预测系统。

This is a complete Python implementation of the Schumann Resonance Remote Sensing mineral prediction system.

## 项目结构 / Project Structure

```
.
└── Python/                    # Python 实现 / Python Implementation
    ├── Core/                  # 核心类 / Core Classes
    │   ├── geo_data_context.py    # 地理数据上下文
    │   ├── fusion_engine.py       # 融合引擎
    │   └── post_processor.py      # 后处理器
    ├── Detectors/             # 探测器 / Detectors
    │   ├── anomaly_detector.py    # 异常探测器基类
    │   ├── red_edge_detector.py   # 红边探测器
    │   ├── intrinsic_detector.py  # 本征吸收探测器
    │   ├── slow_vars_detector.py  # 慢变量探测器
    │   └── known_anomaly_detector.py  # 已知异常探测器
    ├── Utils/                 # 工具类 / Utilities
    │   ├── geo_utils.py          # 地理工具
    │   ├── visualizer.py         # 可视化工具
    │   ├── kmz_mask_generator.py # KMZ掩码生成器
    │   └── export_kmz.py         # KMZ导出
    ├── main.py                # 主程序 / Main Script
    ├── requirements.txt       # Python依赖 / Dependencies
    ├── README.md             # 完整文档 / Full Documentation
    ├── QUICKSTART.md         # 快速开始 / Quick Start
    ├── TESTING_GUIDE.md      # 测试指南 / Testing Guide
    └── CONVERSION_SUMMARY.md # 转换说明 / Conversion Notes
```

## 快速开始 / Quick Start

### 1. 安装依赖 / Install Dependencies

```bash
cd Python
pip install -r requirements.txt
```

### 2. 运行程序 / Run the Program

```bash
python main.py
```

程序会交互式地引导你选择数据文件夹和配置选项。

The program will interactively guide you through selecting data folders and configuration options.

## 系统要求 / Requirements

- **Python**: 3.8+
- **操作系统 / OS**: Windows, Linux, macOS
- **内存 / RAM**: 建议 8GB+ (Recommended 8GB+)
- **数据 / Data**: Sentinel-2, Landsat 8, ASTER 影像

## 主要特性 / Key Features

✅ **完整的Python实现** - Complete Python implementation  
✅ **与MATLAB数学等价** - Mathematically equivalent to MATLAB  
✅ **面向对象设计** - Object-oriented design  
✅ **模块化架构** - Modular architecture  
✅ **交互式界面** - Interactive GUI dialogs  
✅ **KML/KMZ支持** - KML/KMZ support  
✅ **多探测器融合** - Multi-detector fusion  
✅ **自动可视化** - Automatic visualization  

## 文档 / Documentation

详细文档请参考 `Python/` 目录下的文档文件：

For detailed documentation, please refer to the files in the `Python/` directory:

- **README.md** - 完整使用指南 / Complete usage guide
- **QUICKSTART.md** - 5分钟快速开始 / 5-minute quick start
- **TESTING_GUIDE.md** - 测试和验证 / Testing and validation
- **CONVERSION_SUMMARY.md** - 技术转换细节 / Technical conversion details

## 开发者 / Developers

本项目由MATLAB代码转换而来，保持了完全相同的数学逻辑和计算结果。

This project is converted from MATLAB code while maintaining exactly the same mathematical logic and computational results.

## 许可证 / License

请参考项目许可证文件。

Please refer to the project license file.

---

**注意**: 此分支仅包含Python代码。如需查看原始MATLAB代码，请切换到其他分支。

**Note**: This branch contains only Python code. For the original MATLAB code, please switch to other branches.
