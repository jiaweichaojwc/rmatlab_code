import streamlit as st
import os
import sys
import subprocess
import sqlite3
import hashlib

# 必须放在第一行
st.set_page_config(page_title="舒曼波共振遥感预测系统", layout="wide")


# ==========================================
# 数据库与加密辅助函数
# ==========================================
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


def init_db():
    """初始化数据库并注入超级管理员账号"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    conn.commit()

    # 检查是否存在 admin 账号，如果没有则自动创建 (默认密码 admin888)
    c.execute('SELECT * FROM users WHERE username="admin"')
    if not c.fetchone():
        c.execute('INSERT INTO users(username, password) VALUES (?,?)', ('admin', make_hashes('admin888')))
        conn.commit()
    conn.close()


def add_user(username, password):
    """(管理员专用) 向数据库添加新用户"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # 用户名已存在
    finally:
        conn.close()


def get_all_users():
    """(管理员专用) 获取所有普通账号列表"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT username FROM users WHERE username != "admin"')
    data = c.fetchall()
    conn.close()
    return [row[0] for row in data]


def delete_user(username):
    """(管理员专用) 删除账号"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('DELETE FROM users WHERE username=?', (username,))
    conn.commit()
    conn.close()


def login_user(username, password):
    """验证用户名和密码"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data


# ==========================================
# 核心业务逻辑 (带权限隔离)
# ==========================================
def run_main_app():
    try:
        import mineral_core
    except ImportError:
        st.error("⚠️ 未检测到 mineral_core 引擎，请确认是否已在当前环境中安装 setup.py！")
        return

    st.title("🌍 舒曼波共振遥感 - 智能分析系统")
    st.markdown("---")

    with st.sidebar:
        # ====== 身份展示与退出 ======
        current_user = st.session_state['username']
        if current_user == "admin":
            st.success("👑 欢迎回来, **超级管理员 (Admin)**")
        else:
            st.success(f"👋 欢迎回来, **{current_user}**")

        if st.button("🚪 退出登录"):
            st.session_state['logged_in'] = False
            st.rerun()

        # ====== 管理员专属控制台 ======
        if current_user == "admin":
            st.markdown("---")
            with st.expander("🛠️ 管理员控制台 (账号分配)", expanded=False):
                st.markdown("**➕ 创建新账号**")
                new_user = st.text_input("分配用户名", key="new_user_input")
                new_pwd = st.text_input("分配初始密码", key="new_pwd_input")
                if st.button("生成账号"):
                    if new_user and new_pwd:
                        if new_user == "admin":
                            st.error("不能占用 admin 名称！")
                        elif add_user(new_user, make_hashes(new_pwd)):
                            st.success(f"已成功为 【{new_user}】 开通访问权限！")
                        else:
                            st.error("该用户名已存在！")
                    else:
                        st.warning("用户名和密码不能为空")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**🗑️ 账号管理**")
                users_list = get_all_users()
                if users_list:
                    user_to_delete = st.selectbox("选择要注销的账号", [""] + users_list)
                    if st.button("注销该账号") and user_to_delete:
                        delete_user(user_to_delete)
                        st.success(f"已注销账号: {user_to_delete}")
                        st.rerun()
                else:
                    st.info("当前暂无其他普通账号")

        # ====== 遥感算法参数配置 (全员可见) ======
        st.markdown("---")
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
        use_red = st.checkbox("RedEdge (红边)", value=True)
        use_int = st.checkbox("Intrinsic (本征吸收)", value=True)
        use_slow = st.checkbox("SlowVars (慢变量)", value=False)

        st.markdown("---")
        kmz_path = st.text_input("4. KML 已知异常文件 (可选):", value="")
        st.checkbox("KnownAnomaly (KML 异常)", value=bool(kmz_path), disabled=True)

        kmz_threshold = st.slider("5. 生成 KMZ 置信度 (0~1):", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("🚀 开始运行分析", use_container_width=True, type="primary")

    # ====== 核心绘图与运行逻辑 ======
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

            mat_file_path = engine.run_core_algorithm(
                data_dir, roi_file, mineral_type, kmz_path, kmz_threshold,
                bool(use_red), bool(use_int), bool(use_slow)
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


# ==========================================
# 仅保留登录的独立访问入口
# ==========================================
def main():
    init_db()  # 初始化数据库并注入 admin

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        run_main_app()
    else:
        st.markdown("<h1 style='text-align: center;'>🔐 舒曼波共振遥感预测系统</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: gray;'>内部授权访问控制台</h4>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.info("💡 本系统为内部私有部署，不对外开放注册。请使用管理员分配的账号登录。")
            username = st.text_input("👤 用户名")
            password = st.text_input("🔑 密码", type='password')

            if st.button("安全登录", type="primary", use_container_width=True):
                hashed_pswd = make_hashes(password)
                result = login_user(username, check_hashes(password, hashed_pswd))

                if result:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("❌ 用户名或密码错误，或账号尚未开通！")


if __name__ == '__main__':
    main()