import os
import sys
import subprocess
import mineral_core  # 这就是你刚刚成功安装的 MATLAB 核心库！


def main():
    print("==================================================")
    print("      深部矿产预测自动化调度系统 (纯 Python 控制)      ")
    print("==================================================")

    # ---------------- 1. 配置你的找矿任务 ----------------
    # 请把这里的路径替换成你今天想要处理的实际数据路径
    DATA_DIR = r"C:\Users\Deep-Lei\Desktop\data\新疆高昌区库格孜觉北金矿-59.05km2【金】（四川黄金）（20260104任务，20260210下载）\data-矿权"
    ROI_FILE = r"C:\Users\Deep-Lei\Desktop\data\新疆高昌区库格孜觉北金矿-59.05km2【金】（四川黄金）（20260104任务，20260210下载）\经纬度坐标-矿权.xlsx"
    MINERAL_TYPE = "gold"  # 目标矿种

    # KML 已知异常文件路径 (如果这次不想用，就改成空字符串 "")
    KMZ_PATH = r""
    KMZ_THRESHOLD = 0.6

    # ---------------- 2. 唤醒 MATLAB 引擎 ----------------
    print("\n[1/3] 正在启动底层遥感数学引擎...")
    engine = mineral_core.initialize()  # 这一步会在后台静默唤醒 MATLAB Runtime

    print("\n[2/3] 正在执行多源特征提取与融合计算...")
    try:
        # 调用核心算法，它会返回生成好的 .mat 结果文件的绝对路径
        # 你之前在 MATLAB 里写的 run_core_algorithm 现在变成了 Python 函数
        mat_file_path = engine.run_core_algorithm(
            DATA_DIR,
            ROI_FILE,
            MINERAL_TYPE,
            KMZ_PATH,
            KMZ_THRESHOLD
        )
        print(f"✅ 底层计算完成！特征矩阵已保存在: {mat_file_path}")

    except Exception as e:
        print(f"❌ 底层计算出错: {e}")
        engine.terminate()
        sys.exit(1)

    finally:
        # 算完立刻释放内存，好习惯
        engine.terminate()

    # ---------------- 3. 调用 Python 高级制图 ----------------
    print("\n[3/3] 正在唤醒高级学术制图与 KMZ 导出模块...")
    out_dir = os.path.dirname(mat_file_path)

    # 指向你写好的出图脚本 (请确保路径准确无误)
    plot_script = r"C:\Users\Deep-Lei\Desktop\rmatlab_code\Python\utils\chengjie_matlab_code.py"

    if not os.path.exists(plot_script):
        print(f"⚠️ 找不到制图脚本: {plot_script}")
    else:
        # 自动将刚才算出的 .mat 路径传给画图脚本
        cmd = [sys.executable, plot_script, mat_file_path, out_dir]
        subprocess.run(cmd)
        print(f"🎉 全部流水线执行完毕！")
        print(f"📂 最终成果图和 KMZ 叠加包请查看: {out_dir}")


if __name__ == "__main__":
    main()