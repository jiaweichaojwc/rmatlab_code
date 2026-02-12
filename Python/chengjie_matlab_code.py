#承接matlab跑出的程序然后绘制KMZ和将经纬度转化为UTM
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import matplotlib.font_manager as fm
import simplekml
import os
import sys
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev  # 导入样条插值库

# ==================== 0. 核心全局配置（只需修改这里！） ====================
# 基础阈值：控制热力图最小值、红色粗线起始值、透明度计算基准
BASE_THRESHOLD = 0.4
# 重点区域阈值：控制KMZ中导出的等高线最小值
KEY_AREA_THRESHOLD = 0.6
# 基础层级步长：白色细网格线的间隔
LEVEL_STEP = 0.01
# 重点区域层级步长：红色粗线的间隔
HIGH_LEVEL_STEP = 0.01

# ========== 等高线颜色配置接口 ==========
FILL_COLOR = simplekml.Color.yellow  #金色
#FILL_COLOR = "FF3373B8"  # 铜色
#FILL_COLOR = simplekml.Color.black  # 石油黑色和煤
#FILL_COLOR = simplekml.Color.black  # 天然气蓝色
#FILL_COLOR = "FFC0C0D8"  #锡
#FILL_COLOR = "FF8B0000"    #磷
#FILL_COLOR = "FFA02020"  # 铁
#FILL_COLOR = "FF00BFFF"  # 萤石
#FILL_COLOR = "FF808080"
#FILL_COLOR = "FFA8A8C0"  # 锌
#FILL_COLOR = "FF383848"  # 钼
FILL_ALPHA_BASE = 80  # 等高线基础透明度（0-255）
FILL_ALPHA_MAX = 200  # 等高线最大透明度
LINE_COLOR = simplekml.Color.white  # 等高线边线色
LINE_WIDTH = 1  # 等高线边线宽度

# ========== Top整体区域配置（含光滑度配置！） ==========
TOP_AREA_FILL_COLOR = simplekml.Color.purple  # Top整体区域填充色（紫色）
TOP_AREA_FILL_ALPHA = 100  # 区域填充透明度
TOP_AREA_LINE_COLOR = simplekml.Color.white  # 区域边线色
TOP_AREA_LINE_WIDTH = 3  # 区域边线宽度
TOP_AREA_SMOOTH_POINTS = 100
# ===============================================

# ==================== 1. 配置与路径设置 (核心修改点) ====================
# 自动接收来自 MATLAB 的动态路径参数，若无参数则使用默认值
if len(sys.argv) > 2:
    data_path = sys.argv[1]    # 接收 MATLAB 的 dataFile 路径
    output_dir = sys.argv[2]   # 接收 MATLAB 的 outDir 路径
else:
    # 默认路径备用
    data_path = r''
    output_dir = r''

# --- 配置中文字体 ---
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
try:
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except:
    plt.rcParams['font.family'] = ['SimHei', 'Arial', 'sans-serif']

os.makedirs(output_dir, exist_ok=True)

# ==================== 2. 数据读取 ====================
mat_data = sio.loadmat(data_path)
lonGrid = mat_data['lonGrid']
latGrid = mat_data['latGrid']
Au_deep = mat_data['Au_deep']

# --- 读取 ROI (方框区域) ---
has_roi = False
lonROI = np.array([])
latROI = np.array([])
if 'lonROI' in mat_data and 'latROI' in mat_data:
    lonROI = mat_data['lonROI'].flatten()
    latROI = mat_data['latROI'].flatten()
    if len(lonROI) > 0 and len(latROI) > 0:
        has_roi = True

# --- 读取 点位 ---
lonTop = mat_data['lonTop'].flatten() if 'lonTop' in mat_data else np.array([])
latTop = mat_data['latTop'].flatten() if 'latTop' in mat_data else np.array([])
redIdx = mat_data['redIdx'].flatten() if 'redIdx' in mat_data else np.array([])
mineral_type = mat_data['mineral_type'][0] if 'mineral_type' in mat_data else 'gold'

# --- 全局变量定义 ---
n_points = min(10, len(lonTop)) if len(lonTop) > 0 else 0
# 预处理Top点位UTM坐标（后续复用，避免重复计算）
utm_x_Top, utm_y_Top = np.array([]), np.array([])
if n_points > 0:
    avg_lon = np.mean(lonGrid)
    utm_zone = int((avg_lon + 180) / 6) + 1
    crs_utm = f'EPSG:326{utm_zone}'
    temp_transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    utm_x_Top, utm_y_Top = temp_transformer.transform(lonTop[:n_points], latTop[:n_points])


# ==================== 辅助函数：凸包生成 + 样条光滑（核心修改！） ====================
def smooth_polygon(x, y, num_points=100):
    """
    对多边形坐标做B样条插值光滑处理，保持闭合
    :param x: 原始多边形x坐标数组
    :param y: 原始多边形y坐标数组
    :param num_points: 光滑后生成的顶点数，越多越光滑
    :return: 光滑后的x、y坐标数组
    """
    # 移除最后一个点（与第一个点重复，避免插值异常）
    x = x[:-1]
    y = y[:-1]
    # 样条插值（k=3为B样条，保证曲线光滑；per=1表示闭合曲线）
    tck, u = splprep([x, y], k=3, per=True, s=0)
    # 生成新的插值点
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck)
    # 闭合光滑后的多边形
    x_new = np.append(x_new, x_new[0])
    y_new = np.append(y_new, y_new[0])
    return x_new, y_new


def get_top_convex_hull(lon_arr, lat_arr, transformer, smooth_num=TOP_AREA_SMOOTH_POINTS):
    """
    生成包裹所有Top点位的凸包多边形，并做光滑处理
    :param lon_arr: Top点位经度数组
    :param lat_arr: Top点位纬度数组
    :param transformer: 经纬度转UTM的转换器
    :param smooth_num: 光滑后顶点数
    :return: 光滑后的UTM凸包坐标(x,y)、经纬度凸包坐标(lon,lat)
    """
    # 转换为UTM坐标（平面坐标才能计算凸包和插值）
    utm_x, utm_y = transformer.transform(lon_arr, lat_arr)
    points_utm = np.column_stack((utm_x, utm_y))
    # 计算凸包
    hull = ConvexHull(points_utm)
    # 获取凸包顶点的UTM坐标并闭合
    hull_utm_x = points_utm[hull.vertices, 0]
    hull_utm_y = points_utm[hull.vertices, 1]
    hull_utm_x = np.append(hull_utm_x, hull_utm_x[0])
    hull_utm_y = np.append(hull_utm_y, hull_utm_y[0])
    # 对凸包边界做样条光滑处理（核心！）
    hull_utm_x_smooth, hull_utm_y_smooth = smooth_polygon(hull_utm_x, hull_utm_y, smooth_num)
    # 转换回经纬度（用于KMZ）
    hull_lon_smooth, hull_lat_smooth = transformer.transform(hull_utm_x_smooth, hull_utm_y_smooth, direction='INVERSE')
    return (hull_utm_x_smooth, hull_utm_y_smooth), (hull_lon_smooth, hull_lat_smooth)


# ==================== 3. 坐标转换 (全局) ====================
avg_lon = np.mean(lonGrid)
utm_zone = int((avg_lon + 180) / 6) + 1
crs_utm = f'EPSG:326{utm_zone}'
transformer_ll2utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
transformer_utm2ll = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

# 转换网格
utm_x_grid, utm_y_grid = transformer_ll2utm.transform(lonGrid, latGrid)
Au_deep_flip = np.flipud(Au_deep)

# --- 提前转换 ROI 坐标 ---
utm_x_ROI = None
utm_y_ROI = None
if has_roi:
    utm_x_ROI, utm_y_ROI = transformer_ll2utm.transform(lonROI, latROI)


# ==================== 4. 绘图函数（调用光滑后的凸包） ====================
def create_figure(transparent=False):
    fig, ax = plt.subplots(figsize=(14, 11), facecolor='none' if transparent else 'white')

    # 1. 填充色 (热力图)
    contourf = ax.contourf(utm_x_grid, utm_y_grid, Au_deep_flip, 80, cmap='jet', extend='both')
    contourf.set_clim(BASE_THRESHOLD, 1.0)

    # 2. 白色细网格线
    levels = np.arange(BASE_THRESHOLD, 1.0 + LEVEL_STEP, LEVEL_STEP)
    ax.contour(utm_x_grid, utm_y_grid, Au_deep_flip, levels, colors='white', linewidths=0.5)

    # 3. 红色粗线 (重点区域)
    high_levels = np.arange(BASE_THRESHOLD, 1.0 + HIGH_LEVEL_STEP, HIGH_LEVEL_STEP)
    cnt_high = ax.contour(utm_x_grid, utm_y_grid, Au_deep_flip, high_levels, colors='red', linewidths=2.0)

    # === 绘制 ROI 方框 ===
    if has_roi and utm_x_ROI is not None:
        ax.plot(utm_x_ROI, utm_y_ROI, 'k-', linewidth=3, zorder=5)

    # 4. Top 整体光滑区域 + 点位
    global n_points, utm_x_Top, utm_y_Top
    if n_points > 0:
        # 计算光滑后的Top凸包区域
        hull_utm, _ = get_top_convex_hull(lonTop[:n_points], latTop[:n_points], transformer_ll2utm)
        # 绘制Top光滑区域（紫色填充，zorder=9，在点位下方）
        ax.fill(hull_utm[0], hull_utm[1], color='purple', alpha=TOP_AREA_FILL_ALPHA / 255, zorder=9)
        # 绘制Top光滑区域边线（白色粗线）
        ax.plot(hull_utm[0], hull_utm[1], color='white', linewidth=TOP_AREA_LINE_WIDTH, zorder=9)
        # 绘制Top点位（红色，zorder=10，在区域上方）
        ax.scatter(utm_x_Top, utm_y_Top, s=14 ** 2,
                   facecolor='red', edgecolor='black', linewidth=2, zorder=10)

    # 5. 重点点位（yellow+red）
    if len(redIdx) > 0 and n_points > 0:
        redIdx_py = redIdx - 1
        # 过滤有效索引
        redIdx_py = redIdx_py[redIdx_py < n_points]
        ax.scatter(utm_x_Top[redIdx_py], utm_y_Top[redIdx_py], s=24 ** 2,
                   facecolor='yellow', edgecolor='red', linewidth=3, zorder=11)

    if transparent:
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])
    else:
        ax.set_aspect('equal')
        ax.set_title(f'2025 {mineral_type} 矿深部预测（UTM Zone {utm_zone}）', fontsize=20)
        ax.set_xlabel(f'UTM X (m)', fontsize=16)
        ax.set_ylabel(f'UTM Y (m)', fontsize=16)
        cbar = fig.colorbar(contourf, ax=ax, location='bottom', shrink=0.8)
        cbar.ax.tick_params(labelsize=12)

    return fig, ax, cnt_high


# ==================== 5. 交互式显示 (信息框在右下角) ====================
print("正在启动交互式窗口...")
fig_interactive, ax_interactive, _ = create_figure(transparent=False)

# --- 鼠标悬停信息框 ---
info_text = ax_interactive.text(
    0.98, 0.02, '',
    transform=ax_interactive.transAxes,
    fontsize=12,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        alpha=0.95,
        edgecolor='black',
        linewidth=1.5
    )
)
info_text.set_visible(False)


def get_value_at_xy(x, y):
    """根据UTM坐标获取经纬度和置信度"""
    lon, lat = transformer_utm2ll.transform(x, y)
    dx = utm_x_grid - x
    dy = utm_y_grid - y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    min_idx = np.unravel_index(np.argmin(dist), dist.shape)
    value = Au_deep_flip[min_idx]
    return lon, lat, round(value, 4)


def on_hover(event):
    """鼠标悬停显示置信度、坐标"""
    if event.inaxes == ax_interactive and event.xdata is not None:
        lon, lat, value = get_value_at_xy(event.xdata, event.ydata)
        info_text.set_text(
            f'📍 UTM: ({event.xdata:.0f}, {event.ydata:.0f})\n'
            f'🌍 经纬度: ({lon:.6f}, {lat:.6f})\n'
            f'🎯 置信度: {value}'
        )
        info_text.set_visible(True)
    else:
        info_text.set_visible(False)
    fig_interactive.canvas.draw_idle()


# 绑定鼠标悬停事件
fig_interactive.canvas.mpl_connect('motion_notify_event', on_hover)
plt.tight_layout()
plt.show(block=True)

# ==================== 6. 生成 KML/KMZ (同步光滑后的Top区域) ====================
print("正在生成地图叠加文件 (KMZ)...")

# --- 1. 生成带 ROI 的透明图片 ---
fig_kml, ax_kml, cnt_high = create_figure(transparent=True)
img_filename = f"【{mineral_type}】预测图_含边界_步长{LEVEL_STEP}.png"
img_path = os.path.join(output_dir, img_filename)
plt.savefig(img_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.close(fig_kml)

# --- 2. 写入 KML ---
kml = simplekml.Kml()
kml.document.name = f"{mineral_type}矿 - 深部预测 (步长{LEVEL_STEP} | 阈值{KEY_AREA_THRESHOLD})"

# 层1: 热力图
ground = kml.newgroundoverlay(name="1. 成矿置信度热力图")
ground.icon.href = img_filename
ground.latlonbox.north = np.max(latGrid)
ground.latlonbox.south = np.min(latGrid)
ground.latlonbox.east = np.max(lonGrid)
ground.latlonbox.west = np.min(lonGrid)
ground.color = 'CC000000'  # 80%不透明

# 层2: 重点区域矢量 (红线)
fol_zones = kml.newfolder(name=f"2. 重点成矿区 (≥{KEY_AREA_THRESHOLD} | 步长{HIGH_LEVEL_STEP})")
for i, level in enumerate(cnt_high.levels):
    if level >= KEY_AREA_THRESHOLD:
        paths = cnt_high.allsegs[i]
        for path in paths:
            if len(path) > 2:
                lons, lats = transformer_utm2ll.transform(path[:, 0], path[:, 1])
                coords = list(zip(lons, lats))
                pol = fol_zones.newpolygon(name=f"置信度 ≥ {level:.2f}", outerboundaryis=coords)
                alpha = int(FILL_ALPHA_BASE + (level - BASE_THRESHOLD) * 100)
                alpha = min(alpha, FILL_ALPHA_MAX)
                pol.style.polystyle.color = simplekml.Color.changealphaint(alpha, FILL_COLOR)
                pol.style.linestyle.color = LINE_COLOR
                pol.style.linestyle.width = LINE_WIDTH

# 层0: ROI 边界
if has_roi:
    fol_roi = kml.newfolder(name="0. 预测区域边界 (ROI)")
    roi_coords = list(zip(lonROI, latROI))
    if roi_coords[0] != roi_coords[-1]:
        roi_coords.append(roi_coords[0])
    pol = fol_roi.newpolygon(name="预测范围", outerboundaryis=roi_coords)
    pol.style.polystyle.color = simplekml.Color.changealphaint(0, simplekml.Color.white)
    pol.style.linestyle.color = simplekml.Color.black
    pol.style.linestyle.width = 5

# 层3: Top 靶区（同步光滑后的紫色整体区域）
if n_points > 0:
    fol_top = kml.newfolder(name="3. Top 靶区 (紫色光滑整体区域)")
    # 计算光滑后的Top凸包经纬度坐标
    _, hull_ll = get_top_convex_hull(lonTop[:n_points], latTop[:n_points], transformer_ll2utm)
    top_area_coords = list(zip(hull_ll[0], hull_ll[1]))
    # 创建Top光滑多边形区域
    pol = fol_top.newpolygon(name=f"Top靶区光滑整体 (共{n_points}个点位)", outerboundaryis=top_area_coords)
    # 设置紫色区域样式
    pol.style.polystyle.color = simplekml.Color.changealphaint(TOP_AREA_FILL_ALPHA, TOP_AREA_FILL_COLOR)
    pol.style.linestyle.color = TOP_AREA_LINE_COLOR
    pol.style.linestyle.width = TOP_AREA_LINE_WIDTH

    # 可选：保留Top点位中心点标记（便于定位单个点，可注释删除）
    fol_top_points = fol_top.newfolder(name="Top点位中心点")
    for i in range(n_points):
        pnt = fol_top_points.newpoint(name=f"Top_{i + 1}", coords=[(lonTop[i], latTop[i])])
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-circle.png'
        pnt.style.iconstyle.scale = 1.2

# --- 保存 KMZ ---
kmz_path = os.path.join(output_dir, f"【{mineral_type}矿】预测图_含边界_步长{LEVEL_STEP}_阈值{KEY_AREA_THRESHOLD}.kmz")
kml.savekmz(kmz_path)

# ==================== 7. 输出完成信息 ====================
print(f"\n所有文件生成完成！")
print(f"当前配置：")
print(f"基础阈值：{BASE_THRESHOLD} | 重点区域阈值：{KEY_AREA_THRESHOLD}")
print(f"网格步长：{LEVEL_STEP} | 红线步长：{HIGH_LEVEL_STEP}")
print(f"Top光滑区域：紫色填充 (透明度{TOP_AREA_FILL_ALPHA}) | 白色边线 (宽度{TOP_AREA_LINE_WIDTH})")
print(f"边界光滑度：{TOP_AREA_SMOOTH_POINTS}个插值顶点（数值越大越光滑）")
print(f"Top靶区：共{n_points}个点位，已生成光滑凸包整体区域")
print(f"文件路径：")
print(f"透明预测图：{img_path}")
print(f"地图叠加包：{kmz_path}")
print(f"可用Google Earth/奥维地图直接打开KMZ，Top紫色区域为光滑曲线边界！")