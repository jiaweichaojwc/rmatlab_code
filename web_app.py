import streamlit as st
import os
import sys
import subprocess

st.set_page_config(page_title="舒曼波共振遥感预测系统", layout="wide")

try:
    import mineral_core
except ImportError:
    st.error("⚠️ 未检测到 mineral_core 引擎，请确认是否已在当前环境中安装 setup.py！")


def main():
    st.title("🌍 舒曼波共振遥感 - 智能分析系统 (纯血控制版)")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 参数配置")

        data_dir = st.text_input("1. Data 数据文件夹:",
                                 value=r"C:\Users\Deep-Lei\Desktop\data\新疆高昌区库格孜觉北金矿-59.05km2【金】（四川黄金）（20260104任务，20260210下载）\data-矿权")
        roi_file = st.text_input("2. 坐标文件 (.xlsx):",
                                 value=r"C:\Users\Deep-Lei\Desktop\data\新疆高昌区库格孜觉北金矿-59.05km2【金】（四川黄金）（20260104任务，20260210下载）\经纬度坐标-矿权.xlsx")
        mineral_type = st.selectbox("3. 目标矿种:",
                                    ['gold', 'copper', 'cave', 'iron', 'lead', 'zinc', 'petroleum', 'gas', 'rare_earth',
                                     'lithium'])

        st.markdown("---")
        st.markdown("**📌 启用的探测器 (自由控制):**")
        # 自由勾选，且变量会被记录下来传给 MATLAB
        use_red = st.checkbox("RedEdge (红边)", value=True)
        use_int = st.checkbox("Intrinsic (本征吸收)", value=True)
        use_slow = st.checkbox("SlowVars (慢变量)", value=False)

        st.markdown("---")
        kmz_path = st.text_input("4. KML 已知异常文件 (可选):", value="")
        st.checkbox("KnownAnomaly (KML 异常)", value=bool(kmz_path), disabled=True)

        kmz_threshold = st.slider("5. 生成 KMZ 置信度 (0~1):", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
        task_name = st.text_input("6. 任务名称 (可选):", placeholder="例如: 新疆金矿_测试01")

        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("🚀 开始运行分析", use_container_width=True, type="primary")

    tab_log, tab_resonance, tab_fusion, tab_prediction = st.tabs([
        "📝 运行日志", "📊 1. 共振参数", "🧩 2. 掩码集成", "🎯 3. 深部预测"
    ])

    with tab_log:
        status_box = st.empty()
        status_box.info("=== 系统就绪，请配置左侧参数 ===")

    if start_btn:
        if not data_dir or not roi_file:
            st.warning("⚠️ 请先在左侧输入数据文件夹和坐标文件的路径！")
            return

        with tab_log:
            status_box.info("⏳ 正在启动底层遥感数学引擎...")

        try:
            engine = mineral_core.initialize()

            with tab_log:
                status_box.warning("🧠 正在执行多源特征提取与融合 (这可能需要几分钟，请耐心等待)...")

            # 【核心】：真正地把你的勾选状态（True/False）传给 MATLAB！
            mat_file_path = engine.run_core_algorithm(
                data_dir,
                roi_file,
                mineral_type,
                kmz_path,
                kmz_threshold,
                bool(use_red),
                bool(use_int),
                bool(use_slow)
            )

            with tab_log:
                status_box.success(f"✅ 底层计算完成！特征矩阵保存在: {mat_file_path}")
                status_box.info("🎨 正在唤醒高级学术制图与 KMZ 导出模块...")

            engine.terminate()

            out_dir = os.path.dirname(mat_file_path)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            plot_script = os.path.join(current_dir, "utils", "chengjie_matlab_code.py")

            if os.path.exists(plot_script):
                cmd = [sys.executable, plot_script, mat_file_path, out_dir]
                subprocess.run(cmd)
                with tab_log:
                    status_box.success(f"🎉 全部流水线执行完毕！结果已保存在: {out_dir}")
            else:
                with tab_log:
                    status_box.error(f"⚠️ 找不到制图脚本: {plot_script}")
                return

            img1_path = os.path.join(out_dir, "01_共振参数综合图.png")
            img3_path = os.path.join(out_dir, "03_深部成矿预测图.png")

            if os.path.exists(img1_path):
                with tab_resonance:
                    st.image(img1_path, use_container_width=True)
            else:
                with tab_resonance:
                    st.info("未生成共振参数图")

            fusion_imgs = [f for f in os.listdir(out_dir) if f.startswith("02_掩码集成") and f.endswith(".png")]
            if fusion_imgs:
                with tab_fusion:
                    st.image(os.path.join(out_dir, fusion_imgs[0]), use_container_width=True)
            else:
                with tab_fusion:
                    st.info("未生成掩码集成图")

            if os.path.exists(img3_path):
                with tab_prediction:
                    st.image(img3_path, use_container_width=True)
            else:
                with tab_prediction:
                    st.info("未生成深部预测图")

        except Exception as e:
            with tab_log:
                st.error(f"❌ 运行发生严重错误:\n{str(e)}")
            try:
                engine.terminate()
            except:
                pass


if __name__ == "__main__":
    main()