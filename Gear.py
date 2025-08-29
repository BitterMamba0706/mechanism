# gear_involute_full_with_involute_lines.py
import streamlit as st
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib import font_manager
import pandas as pd
import io

# ---------- 中文字体检测 ----------
def set_chinese_font():
    common_names = [
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC", "AR PL UKai CN"
    ]
    for name in common_names:
        try:
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            _ = font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None

found_font = set_chinese_font()

st.set_page_config(page_title="完整渐开线齿轮（含啮合线）绘制器", layout="wide")
st.title("完整渐开线直齿圆柱齿轮绘制器（含整圈啮合线）")

st.markdown("""
**说明：** 输入齿数 `z` 与模数 `m` 并可调整齿形系数后，程序会：
- 计算齿轮常用几何参数；
- 绘制完整整圈齿轮轮廓（填充或不填充）；
- 对每一齿分别绘制左右渐开线（啮合线），从基圆展开至齿顶圆；
- 用不同颜色显示：分度圆、基圆、齿顶圆、齿根圆。
""")

# ---------- 输入参数表单 ----------
with st.sidebar.form("params"):
    st.header("齿轮参数")
    z = st.number_input("齿数 z", value=20, min_value=3, step=1)
    m = st.number_input("模数 m (同尺寸单位)", value=2.0, min_value=1e-6, format="%.6f")
    alpha_deg = st.number_input("压力角 α (度)", value=20.0, min_value=5.0, max_value=30.0, step=0.5)
    ha_coef = st.number_input("齿顶系数 ha* (默认=1.0)", value=1.0, min_value=0.0, step=0.1)
    hf_coef = st.number_input("齿根系数 hf* (默认=1.25)", value=1.25, min_value=0.0, step=0.05)
    flank_points = st.slider("单侧渐开线点数（越多越精细）", min_value=50, max_value=2000, value=300, step=10)
    show_fill = st.checkbox("填充齿形（灰色）并显示边界", value=True)
    show_involute_lines = st.checkbox("显示啮合线（渐开线）", value=True)
    involute_line_width = st.slider("啮合线宽度", min_value=0.3, max_value=3.0, value=0.9, step=0.1)
    submitted = st.form_submit_button("绘制完整齿轮")

if not submitted:
    st.info("请在左侧填写参数并点击“绘制完整齿轮”。")
    st.stop()

# ---------- 基本几何量 ----------
alpha = math.radians(alpha_deg)
d = m * z
r = d / 2.0         # 分度圆半径
db = d * math.cos(alpha)
rb = db / 2.0       # 基圆半径
ra = (d + 2.0 * ha_coef * m) / 2.0  # 齿顶圆半径
rf = (d - 2.0 * hf_coef * m) / 2.0  # 齿根圆半径 (可能为负)
pitch = math.pi * m
addendum = ha_coef * m
dedendum = hf_coef * m
whole_depth = addendum + dedendum
tooth_thickness = pitch / 2.0

# 显示参数表
st.subheader("计算参数（几何量）")
params = {
    "齿数 z": z,
    "模数 m": m,
    "压力角 α (°)": alpha_deg,
    "分度圆直径 d": d,
    "分度圆半径 r": r,
    "基圆直径 db": db,
    "基圆半径 rb": rb,
    "齿顶圆半径 ra": ra,
    "齿根圆半径 rf": rf,
    "齿顶高 ha": addendum,
    "齿根高 hf": dedendum,
    "全高 h": whole_depth,
    "圆周节 p": pitch,
    "齿厚 s(分度圆)": tooth_thickness,
}
st.table(pd.DataFrame.from_dict(params, orient="index", columns=["数值"]).style.format("{:.6g}"))

# ---------- 工具函数 ----------
def rotate_pts(pts, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    rot = np.array([[ca, -sa], [sa, ca]])
    return pts.dot(rot.T)

def involute_curve(rb, r_target, n_points):
    """
    从基圆 (t=0) 展开到 r_target 的渐开线点集（本体在 +x 方向）。
    若 r_target <= rb 或 rb<=0 则退化处理返回基圆上的点。
    """
    if rb <= 0 or r_target <= rb:
        return np.array([[rb, 0.0]])
    t_max = math.sqrt((r_target / rb) ** 2 - 1.0)
    t_vals = np.linspace(0.0, t_max, max(2, int(n_points)))
    x = rb * (np.cos(t_vals) + t_vals * np.sin(t_vals))
    y = rb * (np.sin(t_vals) - t_vals * np.cos(t_vals))
    return np.vstack([x, y]).T

# ---------- 生成单齿轮廓（封闭） ----------
base_involute = involute_curve(rb, ra, flank_points)

# 计算分度圆处对应的 t_p 与对应的极角 ang_p，用于将渐开线对齐齿厚中心线
if r <= rb or rb <= 0:
    t_p = 0.0
else:
    t_p = math.sqrt((r / rb) ** 2 - 1.0)

x_p = rb * (math.cos(t_p) + t_p * math.sin(t_p))
y_p = rb * (math.sin(t_p) - t_p * math.cos(t_p))
ang_p = math.atan2(y_p, x_p)
half_tooth_angle = math.pi / z
rot = half_tooth_angle - ang_p

right_flank = rotate_pts(base_involute, rot)      # 右侧渐开线（+y）
left_flank = rotate_pts(base_involute, -rot)      # 左侧原型
left_flank[:,1] *= -1.0                           # 关于 x 轴镜像得到左侧

# 计算基圆处与顶端处的角度（用于构建顶弧与根弧）
ang_base_r = math.atan2(right_flank[0,1], right_flank[0,0])
ang_base_l = math.atan2(left_flank[0,1], left_flank[0,0])
ang_tip_r = math.atan2(right_flank[-1,1], right_flank[-1,0])
ang_tip_l = math.atan2(left_flank[-1,1], left_flank[-1,0])

# 规范化角度为逆时针增加
def ensure_ccw(a1, a2):
    if a2 <= a1:
        return a2 + 2 * math.pi
    return a2

ang_tip_l = ensure_ccw(ang_tip_r, ang_tip_l)
ang_base_l = ensure_ccw(ang_base_r, ang_base_l)

# 顶弧
n_tip = max(8, int(abs(ang_tip_l - ang_tip_r) / 0.005))
tip_angles = np.linspace(ang_tip_r, ang_tip_l, n_tip)
tip_arc = np.vstack([ra * np.cos(tip_angles), ra * np.sin(tip_angles)]).T

# 齿根圆绘图半径（若 rf <=0 使用小正数以便绘制）
rf_plot = rf if rf > 0 else max(0.01 * m, 0.001)
a_br = ang_base_r
a_bl = ang_base_l
if a_bl <= a_br:
    a_bl += 2*math.pi
root_angles = np.linspace(a_br, a_bl, max(8, int((a_bl - a_br)/0.005)))
root_arc = np.vstack([rf_plot * np.cos(root_angles), rf_plot * np.sin(root_angles)]).T

# 组合单齿（严格逆时针闭合）
tooth_coords_list = []
for p in root_arc:
    tooth_coords_list.append(p)
for p in left_flank:
    tooth_coords_list.append(p)
for p in tip_arc:
    tooth_coords_list.append(p)
for p in right_flank[::-1]:
    tooth_coords_list.append(p)
tooth_coords = np.array(tooth_coords_list)
if not np.allclose(tooth_coords[0], tooth_coords[-1]):
    tooth_coords = np.vstack([tooth_coords, tooth_coords[0]])

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(8,8))

# 先绘填充齿形（或仅边界）
patches = []
for k in range(z):
    ang_k = 2.0 * math.pi * k / z
    pts_rot = rotate_pts(tooth_coords, ang_k)
    poly = Polygon(pts_rot, closed=True)
    patches.append(poly)

if show_fill:
    pc = PatchCollection(patches, match_original=False)
    pc.set_facecolor("#D9D9D9")
    pc.set_edgecolor("k")
    pc.set_linewidth(0.35)
    ax.add_collection(pc)
else:
    for p in patches:
        ax.add_patch(p)

# ---------- 在整圈上绘制渐开线（左右侧）作为“啮合线” ----------
if show_involute_lines:
    # 绘制每个齿的右/左渐开线（不裁剪成多边形，仅显示线）
    for k in range(z):
        ang_k = 2.0 * math.pi * k / z
        # 右侧渐开线
        pts_r = rotate_pts(right_flank, ang_k)
        ax.plot(pts_r[:,0], pts_r[:,1], linewidth=involute_line_width, linestyle='-', alpha=0.9, zorder=6, color='tab:orange')
        # 左侧渐开线
        pts_l = rotate_pts(left_flank, ang_k)
        ax.plot(pts_l[:,0], pts_l[:,1], linewidth=involute_line_width, linestyle='-', alpha=0.9, zorder=6, color='tab:orange')

# ---------- 绘制参考圆并用不同颜色区分 ----------
# 分度圆 (蓝)
pitch_circle = Circle((0,0), r, fill=False, linestyle='--', linewidth=1.5)
ax.add_patch(pitch_circle)
# 基圆 (绿)
base_circle = Circle((0,0), rb, fill=False, linestyle='-.', linewidth=1.5)
ax.add_patch(base_circle)
# 齿顶圆 (红)
addendum_circle = Circle((0,0), ra, fill=False, linestyle=':', linewidth=1.5)
ax.add_patch(addendum_circle)
# 齿根圆 (紫)
root_circle = Circle((0,0), rf_plot, fill=False, linestyle='-', linewidth=1.5)
ax.add_patch(root_circle)

# 用颜色手动绘制可视化（因为 Circle 的默认颜色与线型受 patch 控制）
# 绘制显式圈以便指定颜色（用 plot）
theta_full = np.linspace(0, 2*np.pi, 512)
ax.plot(r * np.cos(theta_full), r * np.sin(theta_full), color='tab:blue', linestyle='--', linewidth=1.2, label='分度圆 (r)')
ax.plot(rb * np.cos(theta_full), rb * np.sin(theta_full), color='tab:green', linestyle='-.', linewidth=1.2, label='基圆 (rb)')
ax.plot(ra * np.cos(theta_full), ra * np.sin(theta_full), color='tab:red', linestyle=':', linewidth=1.2, label='齿顶圆 (ra)')
ax.plot(rf_plot * np.cos(theta_full), rf_plot * np.sin(theta_full), color='tab:purple', linestyle='-', linewidth=1.2, label='齿根圆 (rf)')

# ---------- 图形美化 ----------
ax.set_aspect('equal', 'box')
lim = ra * 1.3
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("x (same size unit)")
ax.set_ylabel("y (same size unit)")
ax.set_title(f"Complete involute straight-toothed cylindrical gear (meshing line shown) — z={z}, m={m}, α={alpha_deg}°")
ax.grid(True, linestyle=':', linewidth=0.4)

# 自定义图例：齿形、啮合线、四个参考圆
legend_handles = [
    Line2D([0],[0], color='k', lw=1, label='Tooth profile (polygonal)'),
    Line2D([0],[0], color='tab:orange', lw=1.5, label='Tooth contact line (involute curve)'),
    Line2D([0],[0], color='tab:blue', lw=1.2, linestyle='--', label='Pitch circle (r)'),
    Line2D([0],[0], color='tab:green', lw=1.2, linestyle='-.', label='Base radius (rb)'),
    Line2D([0],[0], color='tab:red', lw=1.2, linestyle=':', label='Tooth tip radius (ra)'),
    Line2D([0],[0], color='tab:purple', lw=1.2, linestyle='-', label='Root radius (rf)')
]
ax.legend(handles=legend_handles, loc='upper right')

st.pyplot(fig)

# ---------- 导出数据 ----------
st.subheader("导出坐标数据")
tooth_df = pd.DataFrame(tooth_coords, columns=["x", "y"])
st.markdown("单齿坐标（齿根右端起，沿逆时针）示例")
st.dataframe(tooth_df.head(200))
buf = io.StringIO()
tooth_df.to_csv(buf, index=False)
st.download_button("下载单齿坐标 CSV（UTF-8）", data=buf.getvalue().encode('utf-8-sig'),
                   file_name=f"tooth_coords_z{z}_m{m}.csv", mime="text/csv")

# 全齿轮点（不包含闭合重复点）
all_coords = []
for k in range(z):
    ang_k = 2.0 * math.pi * k / z
    pts_rot = rotate_pts(tooth_coords[:-1], ang_k)
    for p in pts_rot:
        all_coords.append((k, p[0], p[1]))
all_df = pd.DataFrame(all_coords, columns=["tooth_index", "x", "y"])
buf2 = io.StringIO()
all_df.to_csv(buf2, index=False)
st.download_button("下载整齿轮轮廓点 CSV（UTF-8）", data=buf2.getvalue().encode('utf-8-sig'),
                   file_name=f"gear_outline_z{z}_m{m}.csv", mime="text/csv")

