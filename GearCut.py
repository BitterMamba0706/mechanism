# gear_generation_rack_app.py
"""
渐开线齿轮—齿条插刀（简化三角齿刀）切制 仿真（Streamlit）
- 输入：m, z, alpha (deg), 变位系数 x, 齿轮中心 (x0, y0)
- 动态演示：齿条横向滑动，齿轮按滚动约束旋转；记录刀齿在齿轮坐标下位置，取其包络近似生成齿形。
- 输出：动画 GIF、最终齿廓点 CSV、理论渐开线供比对。
"""
import streamlit as st
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import pandas as pd
import io
import imageio
from matplotlib import font_manager

# ---------- 中文字体检测以避免乱码 ----------
def set_chinese_font():
    common_names = [
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "PingFang SC"
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

st.set_page_config(page_title="齿条插刀齿轮成形（动态仿真）", layout="wide")
st.title("齿条插刀切制齿轮（动态仿真）")

st.markdown(
    "本演示将齿条简化为三角齿刀（等节距三角齿），模拟齿条沿水平方向滑动，齿轮按基圆滚动关系旋转，"
    "通过累积刀齿位置的包络来近似得到生成齿廓（可与理论渐开线比较）。"
)

# ---------- 参数输入 ----------
with st.sidebar.form("params"):
    st.header("输入参数")
    m = st.number_input("模数 m (同尺寸单位，mm)", value=2.0, min_value=1e-6, format="%.6f")
    z = st.number_input("齿数 z", value=24, min_value=4, step=1)
    alpha_deg = st.number_input("压力角 α (度)", value=20.0, min_value=5.0, max_value=30.0, step=0.5)
    x_shift_coef = st.number_input("齿轮变位系数 x (可正可负)", value=0.0, step=0.01, format="%.3f")
    x0 = st.number_input("齿轮中心 x0", value=0.0, format="%.4f")
    y0 = st.number_input("齿轮中心 y0", value=0.0, format="%.4f")

    st.markdown("---")
    st.header("仿真/动画参数")
    frames = st.slider("动画帧数（步数）", min_value=40, max_value=240, value=120, step=10)
    teeth_span = st.slider("齿条覆盖宽度倍数（相对于齿顶圆周）", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    tooth_height_coeff = st.slider("刀齿高度倍数（相对于模数 m）", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
    frame_size = st.selectbox("单帧图像长宽 (像素)", options=[400, 600, 800], index=1)
    show_theory = st.checkbox("显示理论渐开线对比曲线", value=True)
    submitted = st.form_submit_button("开始仿真并生成动画")

if not submitted:
    st.info("在左侧输入参数后点击“开始仿真并生成动画”。")
    st.stop()

# ---------- 计算常用几何量 ----------
alpha = math.radians(alpha_deg)
d = m * z
r_pitch = d / 2.0
db = d * math.cos(alpha)
rb = db / 2.0
# 采用变位系数 x 后的有效分度直径 d_a? （变位影响齿顶与齿根位置）
# 标准：齿顶圆半径 ra = d/2 + ha* m + 2*x*m (等效处理变位对齿顶)
ha_star = 1.0
hf_star = 1.25
addendum = ha_star * m + 2.0 * x_shift_coef * m  # 简化处理：将变位放到齿顶高度影响上 (近似)
dedendum = hf_star * m - 2.0 * x_shift_coef * m  # 近似
ra = r_pitch + addendum
rf = r_pitch - dedendum
pitch = math.pi * m
half_tooth_thickness = pitch / 2.0  # 在分度圆上的弧长对应的半厚 (线性近似)

# ---------- 构造齿条刀具（简化为等距三角齿刀） ----------
# 刀齿参数：周期 p = pitch; 单齿为等腰三角形，顶端朝下（切入齿轮）
p = pitch
tooth_tip_height = tooth_height_coeff * m   # 刀齿高度（从刀条顶线向下）
# rack reference horizontal line (刀顶线) 放在 y = y0 + rb + clearance
y_rack = y0 + rb + 0.0  # 让刀顶线与基圆切线处对齐（基圆接触线在这里）
# 构造若干刀齿（在 x 方向覆盖足够宽的范围）
circumference_est = 2.0 * math.pi * ra
rack_half_width = teeth_span * circumference_est / 2.0
# 生成刀齿中心位置在 [-rack_half_width, rack_half_width]
kmin = int(math.floor((-rack_half_width) / p)) - 2
kmax = int(math.ceil((rack_half_width) / p)) + 2
tooth_centers = np.arange(kmin, kmax+1) * p

# 单个刀齿三角形（相对于中心 x=0）
def single_rack_tooth_polygon(center_x, y_top, p, tip_h):
    half = p / 2.0
    # simple isosceles triangle: left base, tip, right base
    pts = np.array([
        [center_x - half, y_top],
        [center_x, y_top - tip_h],
        [center_x + half, y_top],
    ])
    return pts

rack_teeth = [single_rack_tooth_polygon(cx, y_rack, p, tooth_tip_height) for cx in tooth_centers]

# ---------- 仿真主循环（产生每帧：刀条位移 s -> 齿轮旋转 phi = s / rb） ----------
# 我们让齿条从 s=0 到 s = rb * 2π （使齿轮完成一圈基圆滚动，从而覆盖整圈轮廓）
if rb <= 0:
    st.error("基圆半径 rb 非正，检查输入参数与压力角。")
    st.stop()

s_total = rb * 2.0 * math.pi
s_vals = np.linspace(0.0, s_total, frames)

# we will collect transformed rack vertices (在齿轮坐标系下) 的集合
# 通过累积这些点的投影包络来近似生成齿轮轮廓
all_transformed_points = []

# convenience rotation about gear center
def rotate_points_about_center(pts, center, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    R = np.array([[ca, -sa],[sa, ca]])
    shifted = pts - np.array(center)
    return (shifted.dot(R.T)) + np.array(center)

# accumulate frames for GIF
frames_images = []
dpi = 100
figsize = (frame_size / dpi, frame_size / dpi)

# For stability, precompute rack global vertices (each tooth polygon vertices)
rack_global_polys = rack_teeth  # already in world coords (centered at x positions)

# iterate frames
st.info("正在计算并渲染动画帧 —— 可能需要几秒钟，取决于帧数和分辨率。")
progress = st.progress(0)
for i, s in enumerate(s_vals):
    # rack is shifted horizontally by +s (向右滑动)
    # due to滚动约束，齿轮逆向旋转 phi = s/rb (这里约定方向)
    phi = s / rb  # positive phi rotates gear CCW; we will rotate rack points by -phi about gear center to view在齿轮坐标下
    current_pts = []
    # transform each rack tooth polygon: shift by s in x, then rotate about gear center by -phi (相当于观察齿轮的参考系)
    for poly in rack_global_polys:
        shifted = poly.copy()
        shifted[:,0] += s  # shift horizontally
        # rotate around gear center by -phi to bring到齿轮坐标系
        transformed = rotate_points_about_center(shifted, (x0, y0), -phi)
        current_pts.append(transformed)
        # 收集点集合
        for pt in transformed:
            all_transformed_points.append(pt)
    # 绘制当前帧：齿轮空白（分度圆、基圆、齿顶圆、齿根圆）、当前齿条位置与已经切出的轮廓近似（包络）
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # 1) 绘制齿廓累计包络（用 current approximation）
    # 用 all_transformed_points 的投影方法在很多方向上取最大投影，获得包络点
    pts = np.array(all_transformed_points) if len(all_transformed_points) > 0 else np.zeros((0,2))
    if pts.shape[0] > 10:
        thetas = np.linspace(0, 2.0 * math.pi, 720)  # 720 directions for smooth
        envelope = []
        center = np.array([x0, y0])
        U = np.column_stack([np.cos(thetas), np.sin(thetas)])  # directions
        # compute dot products (pts - center)·u for all
        pc = (pts - center[None,:])  # Nx2
        # compute projection for each direction: max over points of pc·u
        # to save memory compute chunk-wise
        max_proj = []
        for j in range(U.shape[0]):
            u = U[j]
            proj = pc.dot(u)
            max_p = proj.max() if proj.size>0 else (rf if rf>0 else 0.0)
            max_proj.append(max_p)
        max_proj = np.array(max_proj)
        # clip negative radii
        max_proj = np.maximum(max_proj, rf if rf>0 else 0.0)
        envelope_pts = center[None,:] + np.column_stack([max_proj * np.cos(thetas), max_proj * np.sin(thetas)])
        ax.plot(envelope_pts[:,0], envelope_pts[:,1], color='tab:orange', linewidth=2.0, label='已切出轮廓（近似包络）')
    # 2) 绘制齿轮参考圆（分度、基、齿顶、齿根）
    theta_full = np.linspace(0, 2*np.pi, 500)
    ax.plot(x0 + r_pitch * np.cos(theta_full), y0 + r_pitch * np.sin(theta_full),
            color='tab:blue', linestyle='--', linewidth=1.2, label='分度圆')
    ax.plot(x0 + rb * np.cos(theta_full), y0 + rb * np.sin(theta_full),
            color='tab:green', linestyle='-.', linewidth=1.2, label='基圆')
    ax.plot(x0 + ra * np.cos(theta_full), y0 + ra * np.sin(theta_full),
            color='tab:red', linestyle=':', linewidth=1.2, label='齿顶圆')
    rf_plot = rf if rf>0 else max(0.01*m, 0.001)
    ax.plot(x0 + rf_plot * np.cos(theta_full), y0 + rf_plot * np.sin(theta_full),
            color='tab:purple', linestyle='-', linewidth=1.2, label='齿根圆 (绘图值)')
    # 3) 当前齿条（刀具）位置：绘制每个刀齿多边形
    for poly in current_pts:
        ax.add_patch(Polygon(poly, closed=True, facecolor='#BBBBFF', edgecolor='k', alpha=0.6))
    # 4) optional: draw theoretical involute (单齿示意) for comparison
    if show_theory:
        # 理论单齿渐开线（基于基圆 rb），绘制一颗齿的两侧并旋转排列
        def involute_from_rb(rb, r_to, npts=200):
            if r_to <= rb or rb <= 0:
                return np.array([[rb, 0.0]])
            tmax = math.sqrt((r_to / rb)**2 - 1.0)
            t = np.linspace(0.0, tmax, npts)
            x = rb * (np.cos(t) + t * np.sin(t))
            y = rb * (np.sin(t) - t * np.cos(t))
            return np.vstack([x,y]).T
        # draw one tooth's theoretical flanks and replicate around
        base_invol = involute_from_rb(rb, ra, 300)
        # align to half tooth center as earlier in static code:
        # find tp angle at pitch radius
        if r_pitch > rb:
            t_p = math.sqrt((r_pitch / rb)**2 - 1.0)
        else:
            t_p = 0.0
        x_p = rb * (math.cos(t_p) + t_p * math.sin(t_p))
        y_p = rb * (math.sin(t_p) - t_p * math.cos(t_p))
        ang_p = math.atan2(y_p, x_p)
        half_ang = math.pi / z
        rot_align = half_ang - ang_p
        right_flank = rotate_points_about_center(base_invol, (0,0), rot_align)
        left_flank = rotate_points_about_center(base_invol, (0,0), -rot_align)
        left_flank[:,1] *= -1.0
        # replicate to all teeth
        for k in range(z):
            ang_k = 2.0*math.pi*k/z
            rfk_r = rotate_points_about_center(right_flank, (0,0), ang_k) + np.array([x0,y0])
            rfk_l = rotate_points_about_center(left_flank, (0,0), ang_k) + np.array([x0,y0])
            ax.plot(rfk_r[:,0], rfk_r[:,1], color='tab:orange', linewidth=0.7, alpha=0.8)
            ax.plot(rfk_l[:,0], rfk_l[:,1], color='tab:orange', linewidth=0.7, alpha=0.8)
    # finalize axes
    ax.set_aspect('equal', 'box')
    lim = ra * 1.6
    ax.set_xlim(x0 - lim, x0 + lim)
    ax.set_ylim(y0 - lim, y0 + lim)
    ax.axis('off')
    ax.set_title(f"Rack Forming Tooling Simulation: Frame {i+1}/{frames}")
    # save figure to buffer
    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    fig.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = imageio.v2.imread(buf)
    frames_images.append(img)
    progress.progress((i+1)/frames)
# end loop
progress.empty()

# ---------- 生成 GIF 并显示 ----------
st.subheader("生成的动画 (GIF)")
gif_bytes = io.BytesIO()
imageio.mimsave(gif_bytes, frames_images, format='GIF', fps=20)
gif_bytes.seek(0)
st.image(gif_bytes.getvalue(), caption="齿条插刀切制齿轮（近似仿真）", use_container_width=True)
st.download_button("下载动画 GIF", data=gif_bytes.getvalue(), file_name="gear_generation_by_rack.gif", mime="image/gif")

# ---------- 最终包络 / 齿廓导出 ----------
# 使用前面最后一次的 envelope_pts (若存在)，否则给空表
try:
    final_envelope = envelope_pts  # from last iteration
    final_df = pd.DataFrame(final_envelope, columns=["x","y"])
    st.subheader("近似生成齿廓（包络）")
    st.dataframe(final_df.head(200))
    csvbuf = io.StringIO()
    final_df.to_csv(csvbuf, index=False)
    st.download_button("下载近似齿廓点 CSV (UTF-8)", data=csvbuf.getvalue().encode('utf-8-sig'),
                       file_name=f"generated_gear_profile_z{z}_m{m}.csv", mime="text/csv")
except Exception:
    st.warning("未生成有效包络点。")

# ---------- 显示计算参数 ----------
st.subheader("计算参数与说明")
params = {
    "模数 m": m,
    "齿数 z": z,
    "压力角 α (deg)": alpha_deg,
    "分度半径 r (mm)": r_pitch,
    "基圆半径 rb (mm)": rb,
    "齿顶半径 ra (mm, 近似)": ra,
    "齿根半径 rf (mm, 近似)": rf,
    "齿条刀齿高度 (mm)": tooth_tip_height,
    "用于仿真的帧数": frames,
}
st.table(pd.DataFrame.from_dict(params, orient='index', columns=["数值"]).style.format("{:.6g}"))
