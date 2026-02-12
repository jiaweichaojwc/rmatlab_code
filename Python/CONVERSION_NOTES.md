# MATLAB to Python Conversion Documentation

This document describes the conversion of three key MATLAB modules to Python.

## Converted Modules

### 1. PostProcessor (`Core/post_processor.py`)

**Original**: `Core/PostProcessor.m`

**Functionality**:
- Multi-mask integration and enhancement
- Depth and pressure estimation from resonance frequency data
- Surface mineralization potential calculation
- PCA-based spectral anomaly detection
- Integration with visualization and export functions

**Key Features**:
- Static class with `run()` method maintaining original workflow
- Safe result retrieval with default fallbacks
- Gaussian filtering for spatial smoothing
- Mineral-specific enhancement functions (cave, offshore_petroleum, etc.)
- MAT file export for MATLAB compatibility

**Dependencies**:
- numpy, scipy (gaussian_filter)
- scikit-learn (PCA)
- GeoUtils, Visualizer

**Usage Example**:
```python
from Core.post_processor import PostProcessor

# ctx: GeoDataContext instance
# engine: FusionEngine instance with detector results
# final_mask: Integrated anomaly mask
# out_dir: Output directory path

PostProcessor.run(ctx, engine, final_mask, out_dir)
```

---

### 2. Visualizer (`Utils/visualizer.py`)

**Original**: `Utils/Visualizer.m`

**Functionality**:
- Three main visualization methods for mineral detection results
- Publication-quality figures with matplotlib
- Geospatially-aware plotting with coordinate grids

**Methods**:

#### `run_resonance()`
Creates 2×4 grid showing:
- RGB composite
- F-statistic discrimination map
- Red edge position shift
- Moran's I spatial autocorrelation
- Integrated anomaly mask
- Depth estimate (m)
- Pressure gradient (MPa/km)
- Resonance frequency (MHz)

**Output**: `01_共振参数综合图.png` (300 DPI)

#### `run_mask_fusion()`
Creates adaptive grid layout (1-2 rows) showing:
- Individual detector masks (RedEdge, Intrinsic, SlowVars, KnownAnomaly)
- Final integrated mask

**Features**:
- Custom colormap (white → orange-red)
- Smart layout: ≤4 masks in 1 row, >4 masks in 2 rows
- Chinese title support

**Output**: `02_掩码集成_N图.png` (400 DPI)

#### `run_deep_prediction()`
Creates contour map showing:
- Deep mineralization potential (0.4-1.0 scale)
- ROI boundary overlay
- Top 20 anomalies marked
- Highlighted priority targets

**Output**: 
- `03_深部成矿预测图.png` (500 DPI)
- `03_深部成矿预测图.pkl` (figure object for re-editing)

**Dependencies**:
- matplotlib (with Agg backend for non-interactive use)
- numpy

**Usage Example**:
```python
from Utils.visualizer import Visualizer

# Resonance visualization
Visualizer.run_resonance(
    F_map, delta_red, moran, mask, depth, grad_p, freq, rgb,
    out_dir, lon_grid, lat_grid
)

# Mask fusion
masks = [mask1, mask2, mask3, mask4, mask5]
titles = ['Detector 1', 'Detector 2', 'Detector 3', 'Detector 4', 'Final']
Visualizer.run_mask_fusion(masks, titles, lon_grid, lat_grid, out_dir)

# Deep prediction
Visualizer.run_deep_prediction(
    au_deep, lon_grid, lat_grid, lon_roi, lat_roi,
    lon_top, lat_top, red_idx, 'gold', out_dir
)
```

---

### 3. KMZMaskGenerator (`Utils/kmz_mask_generator.py`)

**Original**: `Utils/KMZMaskGenerator.m`

**Functionality**:
- Extract and parse KML/KMZ files
- Filter geometries by keyword matching
- Rasterize polygons and points onto reference TIFF grid
- Generate binary masks for known anomaly areas

**Key Features**:
- Automatic KMZ extraction to temporary directory
- Regex-based KML parsing (no XML library dependency issues)
- Coordinate transformation using rasterio
- Polygon rasterization using scikit-image
- Point buffering with circular disk

**Workflow**:
1. Extract KML from KMZ (if needed)
2. Parse Placemark features matching target keywords
3. Read reference TIFF geospatial information
4. Convert world coordinates to pixel coordinates
5. Rasterize geometries onto mask array
6. Export to MAT and PNG files

**Dependencies**:
- rasterio (geospatial I/O and transforms)
- fiona (geospatial data reading)
- shapely (geometry operations)
- scipy (savemat)
- scikit-image (polygon rasterization)
- PIL (PNG export)

**Usage Example**:
```python
from Utils.kmz_mask_generator import KMZMaskGenerator

# Initialize generator
generator = KMZMaskGenerator(
    kmz_path='path/to/anomalies.kml',
    tif_path='path/to/reference.tif',
    target_keywords=['矿体投影', 'Object ID', 'ZK', '异常'],
    point_radius_pixel=3
)

# Set output path
generator.output_base = 'path/to/output/mask_result'

# Run pipeline and get mask
mask = generator.run()

# Mask is also saved to:
# - mask_result.mat (MATLAB format)
# - mask_result_VisualCheck.png (visual inspection)
```

**Coordinate Systems**:
- Automatically detects geographic (lat/lon) vs projected (UTM) coordinates
- Uses rasterio affine transforms for accurate pixel mapping
- Handles different TIFF spatial reference formats

---

## Key Differences from MATLAB

### 1. Type Hints
All Python functions include type hints for better code clarity:
```python
def run(ctx, engine, final_mask: np.ndarray, out_dir: str) -> None:
```

### 2. Naming Conventions
- MATLAB: `CamelCase` for methods → Python: `snake_case`
- MATLAB: `SpatialRef` → Python: `rasterio.Affine` transform
- Properties remain descriptive but follow Python conventions

### 3. Array Indexing
- MATLAB: 1-indexed → Python: 0-indexed
- MATLAB: `arr(row, col)` → Python: `arr[row, col]`
- MATLAB: `arr(:, 1:3)` → Python: `arr[:, 0:3]`

### 4. NaN Handling
- MATLAB: `isnan()`, `nan` → Python: `np.isnan()`, `np.nan`
- Python: `np.nan_to_num()` for batch replacement

### 5. Image Processing
- MATLAB: `imgaussfilt()` → Python: `scipy.ndimage.gaussian_filter()`
- MATLAB: `poly2mask()` → Python: `skimage.draw.polygon()`
- MATLAB: `imresize()` → Python: `skimage.transform.resize()`

### 6. Statistical Functions
- MATLAB: `pca()` → Python: `sklearn.decomposition.PCA()`
- MATLAB: `mean(..., 'omitnan')` → Python: `np.nanmean()`
- MATLAB: `std(..., 'omitnan')` → Python: `np.nanstd()`

### 7. File I/O
- MATLAB: `save()` → Python: `scipy.io.savemat()`
- MATLAB: `geotiffinfo()` → Python: `rasterio.open()`
- MATLAB: `unzip()` → Python: `zipfile.ZipFile()`

### 8. Visualization
- MATLAB: `figure()`, `imagesc()`, `colormap()` → Python: `matplotlib` equivalents
- MATLAB: `print()` → Python: `plt.savefig()`
- MATLAB: `tiledlayout()` → Python: `plt.subplot()`

---

## Testing

A comprehensive test suite is provided in `test_conversions.py`:

```bash
cd Python
python3 test_conversions.py
```

**Tests Include**:
- Import verification
- PostProcessor safe_get functionality
- Visualizer rendering with dummy data
- KMZMaskGenerator initialization

All tests should pass with output:
```
============================================================
All tests passed! ✓
============================================================
```

---

## Dependencies

Updated `requirements.txt` includes:
- `scikit-learn>=1.0.0` (for PCA)
- All original dependencies maintained

Install with:
```bash
pip install -r requirements.txt
```

---

## Integration

These modules integrate seamlessly with existing Python infrastructure:

1. **GeoDataContext**: Provides input data (ctx object)
2. **FusionEngine**: Provides detector results (engine object)
3. **GeoUtils**: Provides utility functions (thresholds, transforms, etc.)

**Example Workflow**:
```python
from Core.geo_data_context import GeoDataContext
from Core.fusion_engine import FusionEngine
from Core.post_processor import PostProcessor

# Load data
config = {...}
ctx = GeoDataContext(config)
ctx.load_all_data()

# Run detectors
engine = FusionEngine(ctx)
engine.run_all_detectors()

# Get final mask
final_mask = engine.get_final_mask()

# Post-process and visualize
PostProcessor.run(ctx, engine, final_mask, output_dir)
```

---

## Output Files

### PostProcessor
- `{mineral_type}_Result.mat`: Complete results in MATLAB format
- `01_共振参数综合图.png`: Resonance parameter visualization
- `02_掩码集成_5图.png`: Mask fusion visualization
- `03_深部成矿预测图.png`: Deep prediction map
- `03_深部成矿预测图.pkl`: Editable figure object

### KMZMaskGenerator
- `{output_base}.mat`: Binary mask in MATLAB format
- `{output_base}_VisualCheck.png`: Visual verification image

---

## Notes

1. **Matplotlib Backend**: Visualizer uses 'Agg' backend for non-interactive environments
2. **Chinese Characters**: Full UTF-8 support for Chinese titles and labels
3. **Memory Efficiency**: Uses in-place operations where possible
4. **Error Handling**: Graceful fallbacks for missing detector results
5. **Coordinate Systems**: Automatically handles geographic vs projected coordinates

---

## Future Enhancements

Potential improvements:
- [ ] Parallel processing for multiple regions
- [ ] GPU acceleration for large datasets
- [ ] Interactive visualization with plotly
- [ ] Export to additional formats (GeoTIFF, Shapefile)
- [ ] KMZ export implementation (placeholder in PostProcessor)

---

## Authors

Converted from MATLAB to Python - 2026

Original MATLAB code: Deep-Explor Mineral Detection System
