import sys
import os
import subprocess
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit,
                             QPushButton, QComboBox, QCheckBox, QDoubleSpinBox,
                             QTabWidget, QTextEdit, QFileDialog, QMessageBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont


# ========================================================
# 后台计算线程 (防止主界面在计算时卡死)
# ========================================================
class ComputeThread(QThread):
    log_signal = pyqtSignal(str)  # 发送日志的信号
    finished_signal = pyqtSignal(str)  # 计算完成信号，返回输出文件夹

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.log_signal.emit("=== 开始新的分析任务 ===")
            self.log_signal.emit("正在初始化底层 MATLAB 计算引擎...")

            import mineral_core
            engine = mineral_core.initialize()

            self.log_signal.emit(f"目标矿种: {self.params['mineral_type']}")
            self.log_signal.emit("正在执行多源特征提取与融合 (这可能需要几分钟，请耐心等待)...")

            # 1. 调用 MATLAB 核心引擎
            mat_file_path = engine.run_core_algorithm(
                self.params['data_dir'],
                self.params['roi_file'],
                self.params['mineral_type'],
                self.params['kmz_path'],
                self.params['kmz_threshold']
            )
            self.log_signal.emit(f"✅ 底层计算完成！特征矩阵已保存在: {mat_file_path}")
            engine.terminate()

            # 2. 调用 Python 绘图脚本
            self.log_signal.emit("正在唤醒高级学术制图与 KMZ 导出模块...")
            out_dir = os.path.dirname(mat_file_path)

            current_dir = os.path.dirname(os.path.abspath(__file__))
            plot_script = os.path.join(current_dir, "utils", "chengjie_matlab_code.py")

            if os.path.exists(plot_script):
                cmd = [sys.executable, plot_script, mat_file_path, out_dir]
                subprocess.run(cmd)
                self.log_signal.emit("🎉 全部绘图与导出流程执行完毕！")
            else:
                self.log_signal.emit(f"⚠️ 找不到绘图脚本: {plot_script}，跳过绘图。")

            self.finished_signal.emit(out_dir)

        except Exception as e:
            self.log_signal.emit(f"❌ 运行发生严重错误:\n{str(e)}")
            self.finished_signal.emit("")


# ========================================================
# 主窗口界面
# ========================================================
class MineralApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("舒曼波共振遥感 - 智能分析系统 (纯血 Python 版)")
        self.resize(1150, 720)

        self.current_out_dir = ""
        self.initUI()

    def initUI(self):
        # 主布局 (左右分栏)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ============ 左侧面板 (参数配置) ============
        left_panel = QGroupBox("参数配置")
        left_panel.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout()

        # 1. Data 文件夹
        left_layout.addWidget(QLabel("1. Data 数据文件夹:"))
        h1 = QHBoxLayout()
        self.dir_edit = QLineEdit();
        self.dir_edit.setReadOnly(True)
        self.dir_btn = QPushButton("...")
        self.dir_btn.clicked.connect(self.select_data_dir)
        h1.addWidget(self.dir_edit);
        h1.addWidget(self.dir_btn)
        left_layout.addLayout(h1)

        # 2. 坐标文件
        left_layout.addWidget(QLabel("2. 坐标文件 (.xlsx):"))
        h2 = QHBoxLayout()
        self.roi_edit = QLineEdit();
        self.roi_edit.setReadOnly(True)
        self.roi_btn = QPushButton("...")
        self.roi_btn.clicked.connect(self.select_roi_file)
        h2.addWidget(self.roi_edit);
        h2.addWidget(self.roi_btn)
        left_layout.addLayout(h2)

        # 3. 目标矿种
        left_layout.addWidget(QLabel("3. 目标矿种:"))
        self.mineral_combo = QComboBox()
        self.mineral_combo.addItems(
            ['gold', 'copper', 'cave', 'iron', 'lead', 'zinc', 'petroleum', 'gas', 'rare_earth', 'lithium'])
        left_layout.addWidget(self.mineral_combo)

        # KML 配置
        self.kmz_checkbox = QCheckBox("导入 KML/KMZ 已知异常")
        left_layout.addWidget(self.kmz_checkbox)
        h3 = QHBoxLayout()
        self.kmz_edit = QLineEdit();
        self.kmz_edit.setReadOnly(True);
        self.kmz_edit.setPlaceholderText("KML 文件路径...")
        self.kmz_btn = QPushButton("...")
        self.kmz_btn.clicked.connect(self.select_kmz_file)
        h3.addWidget(self.kmz_edit);
        h3.addWidget(self.kmz_btn)
        left_layout.addLayout(h3)

        # 探测器多选框
        det_group = QGroupBox("启用的探测器 (多选)")
        det_layout = QVBoxLayout()
        self.cb_rededge = QCheckBox("RedEdge (红边)");
        self.cb_rededge.setChecked(True)
        self.cb_intrinsic = QCheckBox("Intrinsic (本征吸收)");
        self.cb_intrinsic.setChecked(True)
        self.cb_slowvars = QCheckBox("SlowVars (慢变量)");
        self.cb_slowvars.setChecked(False)
        self.cb_known = QCheckBox("KnownAnomaly (KML)");
        self.cb_known.setChecked(False)
        det_layout.addWidget(self.cb_rededge);
        det_layout.addWidget(self.cb_intrinsic)
        det_layout.addWidget(self.cb_slowvars);
        det_layout.addWidget(self.cb_known)
        det_group.setLayout(det_layout)
        left_layout.addWidget(det_group)

        # 4. 置信度
        left_layout.addWidget(QLabel("4. 生成KMZ置信度 (0~1):"))
        self.kmz_threshold = QDoubleSpinBox()
        self.kmz_threshold.setRange(0.1, 1.0);
        self.kmz_threshold.setSingleStep(0.05);
        self.kmz_threshold.setValue(0.6)
        left_layout.addWidget(self.kmz_threshold)

        # 5. 任务名称
        left_layout.addWidget(QLabel("5. 任务名称 (可选，留空则自动命名):"))
        self.task_name = QLineEdit();
        self.task_name.setPlaceholderText("例如: 新疆金矿_测试01")
        left_layout.addWidget(self.task_name)

        left_layout.addSpacing(20)

        # 运行按钮
        self.run_btn = QPushButton("开始运行分析")
        self.run_btn.setStyleSheet(
            "background-color: #2E8B57; color: white; font-weight: bold; font-size: 16px; padding: 10px;")
        self.run_btn.clicked.connect(self.run_analysis)
        left_layout.addWidget(self.run_btn)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # ============ 右侧面板 (结果与日志展示) ============
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        # 标签页 1: 日志
        self.log_area = QTextEdit();
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("font-family: Consolas; font-size: 13px; background-color: #f8f9fa;")
        self.tabs.addTab(self.log_area, "运行日志")
        self.log_message("=== 系统就绪，请配置左侧参数 ===")

        # 标签页 2, 3, 4: 结果图片
        self.img_resonance = QLabel("运行后在此显示图片...");
        self.img_resonance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_fusion = QLabel("运行后在此显示图片...");
        self.img_fusion.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_prediction = QLabel("运行后在此显示图片...");
        self.img_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.tabs.addTab(self.img_resonance, "1. 共振参数")
        self.tabs.addTab(self.img_fusion, "2. 掩码集成")
        self.tabs.addTab(self.img_prediction, "3. 深部预测")

        main_layout.addWidget(right_panel)

    # ============ 回调函数 ============
    def log_message(self, msg):
        current_time = time.strftime("%H:%M:%S")
        self.log_area.append(f"[{current_time}] {msg}")

    def select_data_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 Data 数据文件夹")
        if folder: self.dir_edit.setText(folder)

    def select_roi_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择坐标文件", "", "Excel Files (*.xlsx *.xls *.csv)")
        if file: self.roi_edit.setText(file)

    def select_kmz_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择已知异常 KML/KMZ", "", "Google Earth Files (*.kml *.kmz)")
        if file:
            self.kmz_edit.setText(file)
            self.kmz_checkbox.setChecked(True)
            self.cb_known.setChecked(True)

    def run_analysis(self):
        # 参数校验
        if not self.dir_edit.text() or not self.roi_edit.text():
            QMessageBox.warning(self, "配置错误", "请先选择 Data 数据文件夹和坐标文件！")
            return

        # 锁定 UI
        self.run_btn.setEnabled(False)
        self.run_btn.setText("正在计算中...")
        self.run_btn.setStyleSheet(
            "background-color: #808080; color: white; font-weight: bold; font-size: 16px; padding: 10px;")
        self.tabs.setCurrentIndex(0)  # 切回日志页
        self.img_resonance.clear();
        self.img_fusion.clear();
        self.img_prediction.clear()

        # 准备传给后台的参数
        kmz_path = self.kmz_edit.text() if self.kmz_checkbox.isChecked() else ""

        params = {
            'data_dir': self.dir_edit.text(),
            'roi_file': self.roi_edit.text(),
            'mineral_type': self.mineral_combo.currentText(),
            'kmz_path': kmz_path,
            'kmz_threshold': self.kmz_threshold.value()
        }

        # 启动后台线程
        self.thread = ComputeThread(params)
        self.thread.log_signal.connect(self.log_message)
        self.thread.finished_signal.connect(self.on_analysis_finished)
        self.thread.start()

    def on_analysis_finished(self, out_dir):
        # 恢复 UI 状态
        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始运行分析")
        self.run_btn.setStyleSheet(
            "background-color: #2E8B57; color: white; font-weight: bold; font-size: 16px; padding: 10px;")

        if not out_dir: return

        # 尝试加载生成的图片展示在右侧
        img1_path = os.path.join(out_dir, "01_共振参数综合图.png")
        img3_path = os.path.join(out_dir, "03_深部成矿预测图.png")

        def load_img(label, path):
            if os.path.exists(path):
                pixmap = QPixmap(path)
                # 等比例缩放适应窗口
                label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))

        load_img(self.img_resonance, img1_path)
        load_img(self.img_prediction, img3_path)

        # 如果生成了深部预测图，自动跳转到那个 Tab
        if os.path.exists(img3_path):
            self.tabs.setCurrentIndex(3)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MineralApp()
    window.show()
    sys.exit(app.exec())