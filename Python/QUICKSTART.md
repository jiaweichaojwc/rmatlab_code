# Quick Start Guide

## 5-Minute Setup

### 1. Install Python Dependencies

```bash
cd Python
pip install numpy scipy rasterio shapely pandas matplotlib scikit-learn scikit-image simplekml
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a folder with:
- **Sentinel-2 L2A** imagery (B02, B03, B04, B05, B06, B07, B08, B11, B12)
- **Landsat 8** imagery (B2-B7)
- **ASTER** imagery (B01-B14)
- **DEM.tif** (Digital Elevation Model)
- **coordinates.xlsx** or **.csv** (ROI boundary with lon/lat)

### 3. Run the System

```bash
python main.py
```

The system will prompt you to:
1. Select the data folder
2. Select the coordinate file
3. Choose whether to import KML/KMZ known anomalies

### 4. View Results

Results are saved in a timestamped folder:
```
[Detectors]_Result_[mineral]_[timestamp]/
├── 01_共振参数综合图.png
├── 02_掩码集成_N图.png
├── 03_深部成矿预测图.png
└── [mineral]_Result.npz
```

## What Gets Computed

1. **Red Edge Anomaly**: Spectral position shift detection
2. **Intrinsic Absorption**: Mineral absorption features
3. **Slow Variables**: Geological parameter changes (optional)
4. **Known Anomalies**: Import from KML/KMZ (optional)
5. **Fusion**: Combined multi-layer anomaly map
6. **Deep Prediction**: Mineral prospectivity at depth

## Key Features

- ✅ Interactive file selection dialogs
- ✅ Automatic coordinate detection from Excel/CSV
- ✅ Multi-detector fusion (OR logic)
- ✅ KML/KMZ import for known deposits
- ✅ Comprehensive visualization
- ✅ Google Earth KMZ export

## Mineral Types Supported

Configure in `main.py`:
```python
config = {
    'mineral_type': 'gold',  # or 'copper', 'cave', etc.
    ...
}
```

## Troubleshooting

**Issue:** "No module named 'tkinter'"
```bash
# Linux
sudo apt-get install python3-tk
```

**Issue:** Memory error
- Reduce image resolution or process smaller areas

**Issue:** Different results from MATLAB
- Check array indexing (Python is 0-based)
- Verify input data format and coordinate system

## Documentation

- **README.md** - Full documentation
- **CONVERSION_SUMMARY.md** - Technical conversion details
- **TESTING_GUIDE.md** - Comprehensive testing procedures

## Support

For issues or questions:
1. Check TESTING_GUIDE.md for troubleshooting
2. Verify your data meets format requirements
3. Compare with MATLAB version if available

## Citation

If using in research, please cite the original MATLAB version and note Python conversion.

---

**Ready to explore!** 🔍🌍✨
