# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import io, math, os

# ---------- 页面配置 ----------
st.set_page_config(page_title="渐开线绘制器 & 函数表", layout="wide")
st.title("渐开线曲线绘制器与渐开线函数表")

# ---------- 尝试设置 matplotlib 中文字体 ----------
def set_chinese_font():
    """
    尝试在系统中查找常见中文字体并设置给 matplotlib。
    返回找到的字体名称（字符串），未找到则返回 None。
    """
    common_names = [
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC", "AR PL UKai CN"
    ]
    # 遍历系统字体文件，检查字体名称中是否包含常见名字
    for font_path in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
        try:
            fp = font_manager.FontProperties(fname=font_path)
            name = fp.get_name()
            for cn in common_names:
                if cn.lower() in name.lower():
                    matplotlib.rcParams['font.family'] = 'sans-serif'
                    matplotlib.rcParams['font.sans-serif'] = [name]
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    return name
        except Exception:
            continue
    # 如果没有在文件名中直接匹配常用名，再尝试直接使用常用名（若系统可识别）
    for name in common_names:
        try:
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            # 测试能否找到字体文件
            _ = font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    # 未找到合适中文字体
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None

found_font = set_chinese_font()

# ---------- 说明 ----------
st.markdown(
    """
**说明：**
- 本工具绘制基圆与渐开线，生成渐开线点数据；同时生成渐开线函数表 `inv(α) = tan(α) - α`。
- 角度输入均为**度**（°），内部会转换为弧度计算。
- 导出的 CSV 使用 `UTF-8 with BOM` (`utf-8-sig`)，以便在 Excel 中正确显示中文。
"""
)

# ---------- 输入参数 ----------
st.sidebar.header("输入参数（角度以度为单位）")
rb = st.sidebar.number_input("基圆半径 rb", value=10.0, min_value=1e-6, format="%.6f")
theta1 = st.sidebar.number_input("渐开线参数 t 起始 θ1 (度)", value=0.0, step=1.0)
theta2 = st.sidebar.number_input("渐开线参数 t 结束 θ2 (度)", value=60.0, step=1.0)
num_points = st.sidebar.slider("曲线点数（越大越平滑）", min_value=50, max_value=5000, value=800, step=50)

st.sidebar.markdown("---")
st.sidebar.header("渐开线函数表（压力角）")
theta3 = st.sidebar.number_input("压力角 起始 θ3 (度)", value=10.0, step=1.0)
theta4 = st.sidebar.number_input("压力角 结束 θ4 (度)", value=30.0, step=1.0)
theta5 = st.sidebar.number_input("压力角 步长 θ5 (度)", value=1.0, min_value=1e-6, step=0.1)

# 输入验证
if theta1 == theta2:
    st.error("theta1 与 theta2 不能相等，请调整。")
    st.stop()
if num_points <= 1:
    st.error("点数必须大于 1。")
    st.stop()
if theta3 > theta4:
    st.error("theta3 必须小于或等于 theta4。")
    st.stop()
if theta5 <= 0:
    st.error("theta5 必须大于 0。")
    st.stop()

# ---------- 计算渐开线点 ----------
t1 = math.radians(theta1)
t2 = math.radians(theta2)
t = np.linspace(t1, t2, int(num_points))

# 渐开线参数方程（基圆半径 rb）
# x = rb*(cos t + t*sin t)
# y = rb*(sin t - t*cos t)
x = rb * (np.cos(t) + t * np.sin(t))
y = rb * (np.sin(t) - t * np.cos(t))

# 基圆用于参考绘制
circle_theta = np.linspace(0, 2 * np.pi, 400)
circle_x = rb * np.cos(circle_theta)
circle_y = rb * np.sin(circle_theta)

# ---------- 绘图：基圆 + 渐开线 ----------
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(circle_x, circle_y, linestyle='--', linewidth=1, label=f"Base circle (rb={rb})")
ax.plot(x, y, linewidth=2, label=f"involute curve: t={theta1}° → {theta2}°")
ax.set_aspect('equal', 'box')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Base circle and involute curve")
ax.legend()
st.pyplot(fig)

# ---------- 显示渐开线点表格 ----------
curve_df = pd.DataFrame({
    "t(rad)": t,
    "t(deg)": np.degrees(t),
    "x": x,
    "y": y,
    "r = sqrt(x^2+y^2)": np.sqrt(x**2 + y**2)
})
with st.expander("查看渐开线点数据（表格）", expanded=False):
    st.dataframe(curve_df.head(200))
    st.write(f"总点数：{len(curve_df)}")

# 下载渐开线点 CSV（utf-8-sig）
csv_buf = io.StringIO()
curve_df.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode('utf-8-sig')
st.download_button("下载渐开线点 CSV（UTF-8-BOM）", data=csv_bytes,
                   file_name="involute_curve_points.csv", mime="text/csv")

# ---------- 生成渐开线函数表（inv(α)） ----------
alphas_deg = np.arange(theta3, theta4 + 1e-9, theta5)
alphas_rad = np.radians(alphas_deg)

inv_values = []
for a in alphas_rad:
    # tan 在接近 90° 时会发散，检查有限性
    if np.isfinite(np.tan(a)):
        inv_values.append(np.tan(a) - a)
    else:
        inv_values.append(np.nan)

inv_df = pd.DataFrame({
    "alpha(deg)": np.round(alphas_deg, 8),
    "alpha(rad)": np.round(alphas_rad, 12),
    "inv(alpha) = tan(alpha) - alpha": np.array(inv_values, dtype=float)
})

st.markdown("### Involute function table（inv(α) = tanα − α）")
st.dataframe(inv_df)

# 绘制 inv(alpha)
fig2, ax2 = plt.subplots(figsize=(6,3.5))
ax2.plot(inv_df["alpha(deg)"], inv_df["inv(alpha) = tan(alpha) - alpha"], marker='o', linewidth=1)
ax2.set_xlabel("Angle of pressure α (degrees)")
ax2.set_ylabel("inv(α) = tanα − α (radians)")
ax2.set_title("The involute function inv(α) varies with the pressure angle α.")
ax2.grid(True)
st.pyplot(fig2)

# 下载 inv 表 CSV（utf-8-sig）
buf2 = io.StringIO()
inv_df.to_csv(buf2, index=False)
st.download_button("下载渐开线函数表 CSV（UTF-8-BOM）", data=buf2.getvalue().encode('utf-8-sig'),
                   file_name="involute_function_table.csv", mime="text/csv")
