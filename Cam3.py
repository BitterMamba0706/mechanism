# -*- coding: utf-8 -*-
"""
swing_roller_cam_app.py

摆动滚子推杆盘形凸轮机构 — Streamlit GUI

保存为 swing_roller_cam_app.py 后运行：
    pip install streamlit numpy matplotlib pandas
    streamlit run swing_roller_cam_app.py
"""
import streamlit as st
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

# ----------- 设置中文字体 ----------- #
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 页面设置
st.set_page_config(page_title="摆动滚子推杆盘形凸轮机构", layout="wide")
st.title("摆动滚子推杆盘形凸轮机构 — 设计与仿真")

# ---------------- Sidebar 输入（都带唯一 key） ----------------
st.sidebar.header("几何与运动参数")
a = st.sidebar.number_input("中心距 a (mm) — 摆轴到凸轮中心的水平距离", value=30.0, format="%.3f", key="a_input")
l = st.sidebar.number_input("摆杆长度 l (mm)", value=50.0, format="%.3f", key="l_input")
r0 = st.sidebar.number_input("基圆半径 r0 (mm)", value=15.0, format="%.3f", key="r0_input")
r1 = st.sidebar.number_input("滚子半径 r1 (mm)", value=6.0, format="%.3f", key="r1_input")

st.sidebar.markdown("---")
st.sidebar.markdown("**运动分段（度）**：θ1（上摆）、θ2（回摆），两者之和按比例归一为360°")
theta1 = st.sidebar.number_input("θ1 上摆段 (°)", value=120.0, min_value=0.1, format="%.3f", key="theta1_input")
theta2 = st.sidebar.number_input("θ2 回摆段 (°)", value=240.0, min_value=0.1, format="%.3f", key="theta2_input")
alpha1_deg = st.sidebar.number_input("上摆角 α1 (°)", value=25.0, format="%.3f", key="alpha1_input")

rotation_dir = st.sidebar.selectbox("旋转方向", ["逆时针", "顺时针"], key="rot_input")
pts = st.sidebar.slider("采样点数 (越大轮廓越平滑)", min_value=400, max_value=8000, value=2400, step=100, key="pts_slider")

st.sidebar.markdown("---")
st.sidebar.caption("注意：若 θ1+θ2 ≠ 360°，程序会按比例缩放二者使和为 360°。")

# 校正角度和
sum_theta = float(theta1) + float(theta2)
if abs(sum_theta - 360.0) > 1e-9:
    st.warning(f"θ1 + θ2 = {sum_theta:.4f}°，程序将按比例缩放两段角度使之和为 360°。")
    scale = 360.0 / max(sum_theta, 1e-9)
    theta1 *= scale
    theta2 *= scale

# ---------------- 运动生成（返回摆角及导数） ----------------
@st.cache_data
def build_swing_motion(theta1_deg, theta2_deg, alpha1_deg, n_pts):
    """
    返回：
      th_deg [N] : 0..360
      phi_rad [N] : 摆杆相对基准(-pi/2)的摆动增量（rad），范围 0..alpha
      dphi_dth_rad [N] : dφ/dθ (rad per deg)
      d2phi_dth2_rad [N] : d²φ/dθ²
    运动律：
      上摆段（0..θ1）: phi = alpha * 0.5 * (1 - cos(pi * tau)), tau = θ/θ1
      回摆段（θ1..360）: phi = alpha * (1 - tau + (1/pi)*sin(pi*tau)), tau = (θ-θ1)/θ2
    """
    th = np.linspace(0.0, 360.0, n_pts)
    phi = np.zeros_like(th)           # rad
    dphi_dth = np.zeros_like(th)      # rad per deg
    d2phi_dth2 = np.zeros_like(th)    # rad per deg^2

    alpha = math.radians(alpha1_deg)
    t1 = float(theta1_deg)
    t2 = float(theta2_deg)

    # 上摆段
    idx_up = (th >= 0.0) & (th <= t1)
    if np.any(idx_up):
        tau = (th[idx_up] - 0.0) / max(t1, 1e-9)
        phi[idx_up] = alpha * 0.5 * (1.0 - np.cos(np.pi * tau))
        dphi_dtau = alpha * 0.5 * np.pi * np.sin(np.pi * tau)
        dphi_dth[idx_up] = dphi_dtau / max(t1, 1e-9)
        d2phi_dtau2 = alpha * 0.5 * (np.pi**2) * np.cos(np.pi * tau)
        d2phi_dth2[idx_up] = d2phi_dtau2 / (max(t1, 1e-9)**2)

    # 回摆段
    idx_ret = (th > t1) & (th <= 360.0)
    if np.any(idx_ret):
        tau = (th[idx_ret] - t1) / max(t2, 1e-9)
        phi[idx_ret] = alpha * (1.0 - tau + (1.0 / np.pi) * np.sin(np.pi * tau))
        dphi_dtau = alpha * (-1.0 + np.cos(np.pi * tau))
        dphi_dth[idx_ret] = dphi_dtau / max(t2, 1e-9)
        d2phi_dtau2 = alpha * (-np.pi * np.sin(np.pi * tau))
        d2phi_dth2[idx_ret] = d2phi_dtau2 / (max(t2, 1e-9)**2)

    # 闭合端点
    phi[-1] = phi[0]
    dphi_dth[-1] = dphi_dth[0]
    d2phi_dth2[-1] = d2phi_dth2[0]

    return th, phi, dphi_dth, d2phi_dth2

th_deg, phi_rad, dphi_dth_rad, d2phi_dth2_rad = build_swing_motion(theta1, theta2, alpha1_deg, pts)

# 旋转方向处理：生成用于旋转的角度序列（deg），以便把全局坐标旋入 cam-fixed
theta_for_rot_deg = th_deg.copy() if rotation_dir == "逆时针" else (-th_deg) % 360.0

# ---------------- 轮廓生成（数值） ----------------
@st.cache_data
def gen_cam_profile_swing(th_deg, theta_for_rot_deg, phi_rad, a, l, r1, r0):
    """
    计算凸轮轮廓（与滚子相切）
    """
    N = len(th_deg)
    pitch = np.zeros((N, 2))
    profile = np.zeros((N, 2))
    bad = 0

    pivot_x = float(a)
    pivot_y = 0.0

    for i in range(N):
        # 摆杆总角
        phi_tot = -0.5 * math.pi + float(phi_rad[i])
        pgx = pivot_x + l * math.cos(phi_tot)
        pgy = pivot_y + l * math.sin(phi_tot)

        # 旋转到 cam-fixed
        thr = math.radians(theta_for_rot_deg[i])
        c = math.cos(thr); s = math.sin(thr)
        pcx = c * pgx + s * pgy
        pcy = -s * pgx + c * pgy
        pitch[i, :] = [pcx, pcy]

        # 滚子相切点 = 滚子中心向凸轮心缩短 r1
        R = math.hypot(pcx, pcy)
        if R <= r1:
            bad += 1
            R = r1 + 1e-6
        unitx, unity = pcx / R, pcy / R
        profile[i, :] = np.array([unitx * (R - r1), unity * (R - r1)])

    return profile, pitch, bad

profile, pitch_curve, bad_count = gen_cam_profile_swing(th_deg, theta_for_rot_deg, phi_rad, a, l, r1, r0)
if bad_count > 0:
    st.info(f"生成轮廓时发现 {bad_count} 个点可能发生下切/干涉（R ≤ r1），已用近似值替代这些点，请调整参数以避免下切。")

# ---------------- 绘图：尽量接近你给出的示意风格（网格、轴范围、多个同心曲线） ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("凸轮轮廓（cam-fixed）与滚子接触示意")
    fig, ax = plt.subplots(figsize=(7,7))

    # 绘制轮廓（外轮廓）
    ax.plot(profile[:, 0], profile[:, 1], '-', linewidth=2.2, label='cam profile')

    # 基圆（显示为中间圆）
    base = Circle((0, 0), r0, fill=False, linestyle='--', linewidth=1.2, label='base circle r0')
    ax.add_patch(base)

    # 把滚子中心轨迹（pitch curve）也画出来
    ax.plot(pitch_curve[:, 0], pitch_curve[:, 1], '-', linewidth=1.0, alpha=0.6, label='roller centers (pitch curve)')

    # 在图上标出圆心 O 和摆轴 pivot
    ax.plot(0, 0, 'ko', markersize=4)
    ax.text(2, 2, 'O', fontsize=10)

    # 绘制查看角度下的滚子和接触点
    view_ang = st.slider("查看角度 θ (deg)", 0.0, 360.0, 0.0, 0.5, key="view_angle")
    idx = int(np.argmin(np.abs(((th_deg - view_ang + 180) % 360) - 180)))
    if 0 <= idx < len(th_deg):
        pc = pitch_curve[idx]
        # 画滚子
        roller = Circle((pc[0], pc[1]), r1, fill=False, edgecolor='tab:orange', linewidth=2.0, label='roller (view)')
        ax.add_patch(roller)
        # 接触点（近似）
        Rcam = math.hypot(pc[0], pc[1])
        if Rcam > 1e-9:
            contact_pt = np.array([pc[0], pc[1]]) * ((Rcam - r1) / Rcam)
            ax.plot([contact_pt[0]], [contact_pt[1]], 'rx', label='approx contact')
            ax.plot([0, pc[0]], [0, pc[1]], '--', alpha=0.6)

        # 显示摆轴与摆杆位置（全局->cam-fixed 投影）
        # 计算全局摆轴和摆杆端点，再旋转到 cam-fixed 同样角度
        phi_tot = -0.5 * math.pi + float(phi_rad[idx])
        pgx = a + l * math.cos(phi_tot)
        pgy = 0.0 + l * math.sin(phi_tot)
        # rotate global pivot and pivot-to-roller into cam-fixed for display of arm
        thr = math.radians(theta_for_rot_deg[idx])
        c = math.cos(thr); s = math.sin(thr)
        pivot_cam_x = c * a + s * 0.0
        pivot_cam_y = -s * a + c * 0.0
        arm_end = np.array([pc[0], pc[1]])
        # draw pivot and arm
        ax.plot(pivot_cam_x, pivot_cam_y, 'ks', markersize=4)
        ax.plot([pivot_cam_x, arm_end[0]], [pivot_cam_y, arm_end[1]], '-', color='tab:green', linewidth=1.6, alpha=0.9, label='swing arm (view)')

    # 画几个参考同心轮廓使图像更像示意图（可选：内圈、外圈）
    # 一个额外外廓，以示不同加工偏置（r0 + some）
    try:
        alt_outer = profile * 1.06
        ax.plot(alt_outer[:,0], alt_outer[:,1], ':', linewidth=1.0, alpha=0.7)
    except Exception:
        pass

    # 设置坐标轴范围与网格（以20mm为网格单位，类似示意图）
    all_x = np.concatenate([profile[:,0], pitch_curve[:,0], np.array([0, a])])
    all_y = np.concatenate([profile[:,1], pitch_curve[:,1], np.array([0, 0])])
    lim = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)), r0 + l + 20)
    # round up to nearest 20* n
    bound = math.ceil((lim + 20) / 20.0) * 20
    ax.set_xlim(-bound, bound*1.1)
    ax.set_ylim(-bound, bound*1.1)

    # customize grid to look like squared grid like in your image
    major = 20
    ax.set_xticks(np.arange(-bound, bound*1.1 + 1, major))
    ax.set_yticks(np.arange(-bound, bound*1.1 + 1, major))
    ax.grid(which='major', linestyle=':', linewidth=0.8)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x / mm')
    ax.set_ylabel('y / mm')
    ax.set_title('Swinging roller push rod disc cam profile')
    ax.legend(loc='upper right')
    st.pyplot(fig)

with col2:
    st.subheader("摆角 / 角速度 / 角加速度（关于 θ）")
    phi_deg = np.degrees(phi_rad)
    dphi_dth_deg = np.degrees(dphi_dth_rad)
    d2phi_dth2_deg = np.degrees(d2phi_dth2_rad)
    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.plot(th_deg, phi_deg, '-', label=r'$\phi(\theta)$ [deg]')
    ax2.plot(th_deg, dphi_dth_deg, '--', label=r'$d\phi/d\theta$ [deg/deg]')
    ax2.plot(th_deg, d2phi_dth2_deg, ':', label=r'$d^2\phi/d\theta^2$ [deg/deg$^2$]')
    ax2.axvline(th_deg[idx], color='k', linewidth=0.8, alpha=0.6)
    ax2.set_xlabel(r'$\theta$ (deg)')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)


# ---------------- 导出 CSV ----------------
st.markdown('---')
st.subheader('导出数据')
if profile is not None and len(profile) > 0:
    df_profile = pd.DataFrame(profile, columns=['x_cam_mm', 'y_cam_mm'])
    df_pitch = pd.DataFrame(pitch_curve, columns=['x_cam_mm', 'y_cam_mm'])
    df_motion = pd.DataFrame({
        'theta_deg': th_deg,
        'phi_deg': phi_deg,
        'dphi_dtheta_deg_per_deg': dphi_dth_deg,
        'd2phi_dtheta2_deg_per_deg2': d2phi_dth2_deg
    })
    st.download_button('下载 凸轮轮廓 CSV', df_profile.to_csv(index=False).encode('utf-8'), 'swing_cam_profile.csv', 'text/csv', key="dl_prof")
    st.download_button('下载 滚子中心轨迹 CSV', df_pitch.to_csv(index=False).encode('utf-8'), 'swing_pitch_curve.csv', 'text/csv', key="dl_pitch")
    st.download_button('下载 运动学 CSV', df_motion.to_csv(index=False).encode('utf-8'), 'swing_motion.csv', 'text/csv', key="dl_motion")