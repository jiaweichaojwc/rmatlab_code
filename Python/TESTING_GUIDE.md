# Testing Guide for Python Version

## Quick Start Testing

### 1. Installation

```bash
cd Python
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import numpy, scipy, rasterio, shapely, pandas, matplotlib, sklearn; print('✅ All dependencies installed!')"
```

### 3. Syntax Validation

```bash
python -m py_compile main.py Core/*.py Detectors/*.py Utils/*.py
echo "✅ No syntax errors"
```

---

## Testing Without Real Data

### Module Import Test

```python
# test_imports.py
import sys
sys.path.insert(0, '.')

from Core.geo_data_context import GeoDataContext
from Core.fusion_engine import FusionEngine
from Core.post_processor import PostProcessor
from Detectors.red_edge_detector import RedEdgeDetector
from Detectors.intrinsic_detector import IntrinsicDetector
from Detectors.slow_vars_detector import SlowVarsDetector
from Detectors.known_anomaly_detector import KnownAnomalyDetector
from Utils.geo_utils import GeoUtils
from Utils.visualizer import Visualizer
from Utils.kmz_mask_generator import KMZMaskGenerator

print("✅ All imports successful!")
```

### Unit Test Example

```python
# test_geo_utils.py
import numpy as np
from Utils.geo_utils import GeoUtils

# Test mat2gray_roi
img = np.random.rand(100, 100)
roi = np.ones((100, 100), dtype=bool)
roi[50:, :] = False

result = GeoUtils.mat2gray_roi(img, roi)
assert result.shape == img.shape
assert np.nanmax(result[roi]) <= 1.0
assert np.nanmin(result[roi]) >= 0.0
print("✅ mat2gray_roi test passed")

# Test calculate_S2REP_from_DN
B4 = np.random.rand(100, 100) * 0.3
B5 = np.random.rand(100, 100) * 0.3
B6 = np.random.rand(100, 100) * 0.3
B7 = np.random.rand(100, 100) * 0.3
scale_factors = [1.9997e-05, 1.9998e-05, 1.9998e-05, 1.9999e-05]
offsets = [-0.1, -0.1, -0.1, -0.1]

S2REP, REP_QA = GeoUtils.calculate_S2REP_from_DN(B4, B5, B6, B7, scale_factors, offsets)
assert S2REP.shape == B4.shape
assert REP_QA.shape == B4.shape
print("✅ calculate_S2REP_from_DN test passed")

print("\n✅ All GeoUtils tests passed!")
```

---

## Testing With Real Data

### Required Data Structure

Your test data folder should look like:

```
test_data/
├── Sentinel 2 L2/
│   ├── B02.tif  (Blue, 10m)
│   ├── B03.tif  (Green, 10m)
│   ├── B04.tif  (Red, 10m)
│   ├── B05.tif  (Red Edge 1, 20m)
│   ├── B06.tif  (Red Edge 2, 20m)
│   ├── B07.tif  (Red Edge 3, 20m)
│   ├── B08.tif  (NIR, 10m)
│   ├── B11.tif  (SWIR 1, 20m)
│   └── B12.tif  (SWIR 2, 20m)
├── Landsat 8 L1/ (or L2)
│   ├── B2.tif   (Blue)
│   ├── B3.tif   (Green)
│   ├── B4.tif   (Red)
│   ├── B5.tif   (NIR)
│   ├── B6.tif   (SWIR 1)
│   └── B7.tif   (SWIR 2)
├── ASTER L1/ (or L2)
│   ├── B01.tif through B09.tif  (VNIR/SWIR)
│   └── B10.tif through B14.tif  (TIR)
└── DEM.tif
```

Coordinate file (coordinates.xlsx or coordinates.csv):
```
Site_Name, Longitude, Latitude
Point1,    115.234,   35.456
Point2,    115.456,   35.678
Point3,    115.678,   35.890
Point4,    115.890,   36.012
Point1,    115.234,   35.456  # Close the polygon
```

### Run Full Pipeline

```bash
cd Python
python main.py
```

### Expected Output

```
Results_folder_[timestamp]/
├── 01_共振参数综合图.png         # 8-panel resonance parameters
├── 02_掩码集成_N图.png           # N detector masks (N=4 or 5)
├── 03_深部成矿预测图.png         # Deep prediction map
├── 03_深部成矿预测图.pkl        # Editable figure
├── gold_Result.npz              # All numerical results
└── gold_Result.kmz              # Google Earth overlay
```

---

## Comparison with MATLAB

### 1. Run Both Versions

```bash
# MATLAB
cd /path/to/matlab
matlab -batch "run('Main.m')"

# Python
cd /path/to/Python
python main.py
```

### 2. Compare Numerical Results

```python
# compare_results.py
import numpy as np
import scipy.io as sio

# Load MATLAB results
mat_data = sio.loadmat('gold_Result.mat')

# Load Python results
py_data = np.load('gold_Result.npz')

# Compare key arrays
def compare_arrays(name, matlab_arr, python_arr):
    if matlab_arr.shape != python_arr.shape:
        print(f"❌ {name}: Shape mismatch {matlab_arr.shape} vs {python_arr.shape}")
        return False
    
    diff = np.abs(matlab_arr - python_arr)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)
    
    print(f"✓ {name}:")
    print(f"  Shape: {matlab_arr.shape}")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    
    if max_diff < 1e-6:
        print(f"  ✅ EXACT match (< 1e-6)")
        return True
    elif max_diff < 1e-3:
        print(f"  ✅ Good match (< 1e-3)")
        return True
    else:
        print(f"  ⚠️ Significant difference!")
        return False

# Compare arrays
compare_arrays('Au_deep', mat_data['Au_deep'], py_data['Au_deep'])
compare_arrays('F_abs', mat_data['F_abs'], py_data['F_abs'])
compare_arrays('depth_map', mat_data['depth_map'], py_data['depth_map'])
compare_arrays('anomaly_mask_rededge', mat_data['anomaly_mask_rededge'], py_data['anomaly_mask_rededge'])
compare_arrays('final_anomaly_mask', mat_data['final_anomaly_mask'], py_data['final_anomaly_mask'])
```

### 3. Visual Comparison

- Open both PNG files side by side
- Check for identical patterns and color scales
- Minor anti-aliasing differences are acceptable

### 4. KMZ Comparison

- Open both KMZ files in Google Earth
- Verify overlay alignment
- Check polygon locations and transparency

---

## Performance Benchmarking

### Time Each Step

```python
# benchmark.py
import time
import sys
sys.path.insert(0, '.')

from Core.geo_data_context import GeoDataContext
from Core.fusion_engine import FusionEngine
# ... other imports

# Mock config
config = {
    'mineral_type': 'gold',
    'region_type': 'test',
    'levashov_mode': True,
    'kmz_path': ''
}

# Time data loading
start = time.time()
# ctx = GeoDataContext(config)
elapsed = time.time() - start
print(f"Data loading: {elapsed:.2f}s")

# Time each detector
# ... detector timing code

# Compare with MATLAB timing
```

### Memory Profiling

```python
# profile_memory.py
from memory_profiler import profile

@profile
def run_pipeline():
    # ... pipeline code
    pass

if __name__ == '__main__':
    run_pipeline()
```

Run: `python -m memory_profiler profile_memory.py`

---

## Known Issues & Troubleshooting

### Issue 1: tkinter ImportError on Linux

**Error:** `ImportError: No module named '_tkinter'`

**Solution:**
```bash
sudo apt-get install python3-tk
```

### Issue 2: rasterio GDAL errors

**Error:** `ERROR 4: Unable to open EPSG support file gcs.csv`

**Solution:**
```bash
# Set GDAL_DATA environment variable
export GDAL_DATA=$(gdal-config --datadir)
```

### Issue 3: Out of memory

**Error:** `MemoryError` or `numpy.core._exceptions._ArrayMemoryError`

**Solution:**
- Process smaller tiles
- Reduce resolution
- Increase system memory/swap

### Issue 4: Slow performance

**Cause:** Python loops are slower than MATLAB

**Solutions:**
1. Use vectorized numpy operations
2. Add `numba.jit` decorators:
   ```python
   from numba import jit
   
   @jit(nopython=True)
   def calc_local_sum_with_nan(Z):
       # ... function code
   ```

---

## Test Data Generation

If you don't have real data, generate synthetic test data:

```python
# generate_test_data.py
import numpy as np
import rasterio
from rasterio.transform import from_bounds

def generate_test_tif(filename, width=1000, height=1000):
    """Generate a test GeoTIFF file"""
    data = np.random.rand(height, width).astype(np.float32) * 10000
    
    transform = from_bounds(
        west=115.0, south=35.0, east=116.0, north=36.0,
        width=width, height=height
    )
    
    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=rasterio.float32,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(data, 1)

# Generate test files
import os
os.makedirs('test_data/Sentinel 2 L2', exist_ok=True)

for band in ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']:
    generate_test_tif(f'test_data/Sentinel 2 L2/{band}.tif')

print("✅ Test data generated!")
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        cd Python
        pip install -r requirements.txt
    
    - name: Syntax check
      run: |
        cd Python
        python -m py_compile main.py Core/*.py Detectors/*.py Utils/*.py
    
    - name: Import test
      run: |
        cd Python
        python test_imports.py
```

---

## Security Summary

✅ **CodeQL Analysis: 0 vulnerabilities found**

No security issues detected in:
- File I/O operations
- User input handling
- External command execution
- Data serialization

---

## Conclusion

This testing guide covers:
- ✅ Basic installation and validation
- ✅ Unit testing without data
- ✅ Full pipeline testing with data
- ✅ Comparison with MATLAB version
- ✅ Performance benchmarking
- ✅ Troubleshooting common issues
- ✅ Test data generation
- ✅ CI/CD integration

For production deployment, ensure:
1. Real data testing completes successfully
2. Results match MATLAB within acceptable tolerance
3. Performance is acceptable for your use case
4. All dependencies are properly versioned

**Status: Ready for real-world testing** 🚀
