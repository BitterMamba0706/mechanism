# -*- coding: utf-8 -*-
"""
Streamlit 应用：偏置直动滚子推杆盘形凸轮机构设计与仿真
功能：
 - 输入参数：偏距 e，基圆半径 r0，滚子半径 r1
 - 输入四段角度：升程角（rise_deg）、远休角（dwell1_deg）、回程角（return_deg）、近休角（dwell2_deg）
 - 输入升程 h，选择旋转方向（逆时针/顺时针）
 - 画出凸轮轮廓（通过数值方法生成）并对推杆位移随凸轮旋转的运动进行仿真
说明：
 - 采用基于"凸轮固有坐标"（cam-fixed frame）的方法：
     对每个凸轮转角 theta（全局），在全局坐标系中取推杆滚子中心位置 (e, r0 + s(theta))，
     将其旋转到凸轮坐标系（即 cam-fixed frame），记为 p_cam = R(-theta) @ [e, r0 + s]，
     则在凸轮坐标系下，滚子与凸轮接触点在半径方向，轮廓点 = unit(p_cam) * (|p_cam| - r1)。
 - 位移段采用平滑的回转型余弦/回转摆线（cycloidal）段以保证端点速度为零。

使用：
    pip install streamlit numpy matplotlib
    streamlit run streamlit_app.py

"""

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

st.set_page_config(page_title="偏置直动滚子推杆盘形凸轮机构设计与仿真", layout="wide")
st.title("偏置直动滚子推杆盘形凸轮机构 — 设计与仿真")

# ---------- 侧边栏：输入参数 ----------
st.sidebar.header("凸轮参数")
e = st.sidebar.number_input("偏距 e (mm)", value=20.0, format="%.4f")
r0 = st.sidebar.number_input("基圆半径 r0 (mm)", value=25.0, format="%.4f")
r1 = st.sidebar.number_input("滚子半径 r1 (mm)", value=5.0, format="%.4f")

st.sidebar.header("运动分段角度（度）")
rise_deg = st.sidebar.number_input("升程角 (rise, °)", value=90.0, format="%.4f")
dwell1_deg = st.sidebar.number_input("远休角 (dwell1, °)", value=90.0, format="%.4f")
return_deg = st.sidebar.number_input("回程角 (return, °)", value=90.0, format="%.4f")
dwell2_deg = st.sidebar.number_input("近休角 (dwell2, °)", value=90.0, format="%.4f")

h = st.sidebar.number_input("升程 h (mm)", value=15.0, format="%.4f")
rotation_dir = st.sidebar.selectbox("凸轮旋转方向", options=["逆时针", "顺时针"])

st.sidebar.markdown("---")
pts = st.sidebar.slider("轮廓离散点数 (越大曲线越平滑)", min_value=200, max_value=5000, value=1200, step=100)

# ---------- 检查与规范化角度 ----------
total_angle = rise_deg + dwell1_deg + return_deg + dwell2_deg
if abs(total_angle - 360.0) > 1e-6:
    st.warning(f"输入的四段角度和为 {total_angle:.4f}°，不是360°。程序将按比例缩放这四段角度使它们之和为360°。")
    scale = 360.0 / total_angle
    rise_deg *= scale
    dwell1_deg *= scale
    return_deg *= scale
    dwell2_deg *= scale

# ---------- 生成运动规律 ----------
@st.cache_data
def build_motion(rise_deg, dwell1_deg, return_deg, dwell2_deg, h, n_pts=1200):
    # 分段 (度) -> 累计角度
    segs = np.array([rise_deg, dwell1_deg, return_deg, dwell2_deg], dtype=float)
    # 角度数组（全局 theta），从 0 到 360 deg
    theta_deg = np.linspace(0.0, 360.0, n_pts)
    s = np.zeros_like(theta_deg)

    # 辅助：将角度归到段内的局部参数 t in [0, seg]
    cum = np.cumsum(segs)
    start = 0.0
    for i, seg in enumerate(segs):
        end = start + seg
        idx = np.where((theta_deg >= start) & (theta_deg <= end))[0]
        if len(idx) > 0:
            t_rel = (theta_deg[idx] - start) / seg  # 0..1
            if i == 0:  # 升程：cycloidal
                # cycloidal displacement
                s[idx] = h * (t_rel - (1.0/(2*math.pi)) * np.sin(2*math.pi*t_rel))
            elif i == 1:  # 远休：保持 h
                s[idx] = h
            elif i == 2:  # 回程：从 h 回到 0，使用倒向 cycloidal
                # t_rel 0..1 maps to h->0
                s[idx] = h * (1.0 - (t_rel - (1.0/(2*math.pi)) * np.sin(2*math.pi*t_rel)))
            else:  # 近休：保持 0
                s[idx] = 0.0
        start = end
    # 为了闭合端点，确保最后点等于0
    s[-1] = s[0]
    return theta_deg, s

theta_deg, s = build_motion(rise_deg, dwell1_deg, return_deg, dwell2_deg, h, n_pts=pts)

# 如果旋转方向是顺时针，我们把角度反向（顺时针减小角）
if rotation_dir == "顺时针":
    theta_deg = (-theta_deg) % 360.0

# ---------- 生成凸轮轮廓（数值） ----------
@st.cache_data
def generate_cam_profile(theta_deg, s, e, r0, r1):
    # theta_deg: array of global cam angles in degrees
    # s: array of displacements (mm)
    n = len(theta_deg)
    profile = []
    pitch = []  # pitch curve (滚子中心在 cam-fixed 下的坐标)
    bad_count = 0
    for i in range(n):
        theta = math.radians(theta_deg[i])
        # follower 滚子中心在全局坐标（固定不随凸轮旋转）的位置
        # 我们取推杆直线为垂直线 x = e，纵向位置为 r0 + s
        y_global = r0 + s[i]
        p_global = np.array([e, y_global])  # 全局坐标下滚子中心
        # 将滚子中心旋转到凸轮坐标系（cam-fixed）：p_cam = R(-theta) @ p_global
        c = math.cos(theta); si = math.sin(theta)
        x_cam =  c * p_global[0] + si * p_global[1]
        y_cam = -si * p_global[0] + c * p_global[1]
        p_cam = np.array([x_cam, y_cam])
        Rnorm = np.linalg.norm(p_cam)
        if Rnorm <= 1e-6:
            bad_count += 1
            continue
        # 轮廓上的点沿从原点指向滚子中心的方向，半径为 Rnorm - r1
        rad = Rnorm - r1
        if rad <= 0:
            # 滚子中心距离中心小于等于 r1，说明参数可能造成干涉或下切
            # 仍然做近似处理，rad 置为小正数
            rad = 1e-6
            bad_count += 1
        unit = p_cam / Rnorm
        prof_pt = unit * rad
        profile.append(prof_pt)
        pitch.append(p_cam)
    profile = np.array(profile)
    pitch = np.array(pitch)
    return profile, pitch, bad_count

profile, pitch, bad_count = generate_cam_profile(theta_deg, s, e, r0, r1)
if bad_count > 0:
    st.info(f"在生成轮廓时发现 {bad_count} 个点可能发生干涉（滚子中心与中心距离 <= r1），程序已做近似处理。请检查参数以避免下切。")

# ---------- 绘图与交互 ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("凸轮轮廓与局部接触演示")
    fig, ax = plt.subplots(figsize=(7,7))
    # 画凸轮轮廓
    if profile.size > 0:
        ax.plot(profile[:,0], profile[:,1], '-', linewidth=2, label='cam profile (roller offset)')
    # 画基圆
    circle_base = Circle((0,0), r0, fill=False, linestyle='--', label='base circle (r0)')
    ax.add_patch(circle_base)
    # 画滚子路径（pitch curve 在 cam-fixed 下）
    if pitch.size > 0:
        ax.plot(pitch[:,0], pitch[:,1], '.', markersize=1, alpha=0.4, label='pitch curve (roller centers)')
    # 画推杆位置线（在 cam-fixed 下它会旋转）: 我们画一条代表 x' 轴上对应推杆所在的轨迹
    # 画原点
    ax.plot(0,0,'ko')
    # 在选定角度处显示滚子
    angle_slider = st.slider("凸轮角度 (deg) — 查看某一角度下的接触", min_value=0.0, max_value=360.0, value=0.0, step=0.5)
    # 找到最接近的 index
    idx = (np.abs(((theta_deg - angle_slider + 180) % 360) - 180)).argmin()
    if pitch.size > 0 and idx < len(pitch):
        p_cam = pitch[idx]
        # 滚子中心在 cam-fixed 下的位置
        # 画滚子
        roller_center = p_cam
        roller_circle = Circle((roller_center[0], roller_center[1]), r1, fill=False, edgecolor='tab:orange', linewidth=2, label='roller')
        ax.add_patch(roller_circle)
        # 接触点（近似）
        norm_p = np.linalg.norm(p_cam)
        if norm_p > 1e-9:
            contact_pt = (p_cam / norm_p) * (norm_p - r1)
            ax.plot([contact_pt[0]], [contact_pt[1]], 'rx', label='approx contact')
        # 画从原点到滚子中心的连线
        ax.plot([0, roller_center[0]], [0, roller_center[1]], ':', color='gray')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.grid(True)
    ax.legend(loc='upper right')
    st.pyplot(fig)

    st.markdown("**提示：** 上图为凸轮坐标系（cam-fixed），凸轮轮廓为 profile 曲线，基圆用虚线表示。滚子中心轨迹（pitch curve）为灰点，选中角度下用橙色圆表示滚子。")

with col2:
    st.subheader("位移与速度曲线（随凸轮角度）")
    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.plot(theta_deg, s, '-', label='displacement s(θ)')
    # 速度 ds/dθ (单位 mm/deg)，数值微分
    ds_dth = np.gradient(s, theta_deg)
    ax2.plot(theta_deg, ds_dth, '--', label='ds/dθ (mm/deg)')
    ax2.axvline(theta_deg[idx], color='k', linewidth=0.8, alpha=0.6)
    ax2.set_xlabel('θ (deg)')
    ax2.set_ylabel('s (mm) / ds/dθ (mm/deg)')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# ---------- 导出轮廓数据 ----------
st.markdown('---')
if profile.size > 0:
    st.subheader('导出轮廓与轨迹数据')
    import pandas as pd
    df_prof = pd.DataFrame(profile, columns=['x_cam_mm','y_cam_mm'])
    df_pitch = pd.DataFrame(pitch, columns=['x_cam_mm','y_cam_mm'])
    df_theta = pd.DataFrame({'theta_deg': theta_deg, 's_mm': s})

    csv_prof = df_prof.to_csv(index=False).encode('utf-8')
    csv_pitch = df_pitch.to_csv(index=False).encode('utf-8')
    csv_theta = df_theta.to_csv(index=False).encode('utf-8')

    st.download_button('下载凸轮轮廓 CSV (cam-fixed)', data=csv_prof, file_name='cam_profile.csv', mime='text/csv')
    st.download_button('下载滚子中心轨迹 CSV (cam-fixed)', data=csv_pitch, file_name='pitch_curve.csv', mime='text/csv')
    st.download_button('下载位移随角度 CSV', data=csv_theta, file_name='displacement_theta.csv', mime='text/csv')
