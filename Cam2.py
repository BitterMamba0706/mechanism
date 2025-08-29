# -*- coding: utf-8 -*-
"""
Cam2_with_flat.py
对心直动平底/滚子推杆盘形凸轮机构 — Streamlit 应用（支持 Roller / Flat 两种从动件）

- Roller: r(θ) = r0 + s(θ)
- Flat: 通过直线族包络方法计算凸轮轮廓（更精确的平底推杆轮廓）
"""
import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

st.set_page_config(page_title="对心直动平底/滚子推杆盘形凸轮机构", layout="wide")
st.title("对心直动平底/滚子推杆盘形凸轮机构 — 设计与仿真")

# ---------- 侧边栏输入（带唯一 key） ----------
st.sidebar.header("凸轮参数")
r0 = st.sidebar.number_input("基圆半径 r0 (mm)", value=25.0, format="%.4f", key="r0_input")

st.sidebar.markdown("---")
st.sidebar.markdown("**从动件类型**")
follower_type = st.sidebar.selectbox("选择从动件类型", ["roller (圆滚子/点接触)", "flat (平底推杆)"], key="follower_type")

st.sidebar.markdown("---")
st.sidebar.markdown("**运动分段（度）**：θ1（升程）、θ2（回程）、θ3（静止），三者之和应为360°")
theta1 = st.sidebar.number_input("升程角 θ1 (°)", value=90.0, min_value=0.0, format="%.4f", key="theta1_input")
theta2 = st.sidebar.number_input("回程角 θ2 (°)", value=120.0, min_value=0.0, format="%.4f", key="theta2_input")
theta3 = st.sidebar.number_input("静止角 θ3 (°)", value=150.0, min_value=0.0, format="%.4f", key="theta3_input")

h1 = st.sidebar.number_input("升程 h1 (mm)", value=12.0, format="%.4f", key="h1_input")
rotation_dir = st.sidebar.selectbox("旋转方向", ["逆时针", "顺时针"], key="rot_input")
pts = st.sidebar.slider("采样点数（越大曲线越平滑）", 300, 8000, 2000, 100, key="pts_slider")

# 校正角度和（若不等于360则按比例缩放并提示）
sum_ang = theta1 + theta2 + theta3
if abs(sum_ang - 360.0) > 1e-9:
    st.warning(f"θ1+θ2+θ3 = {sum_ang:.4f}°，程序将按比例缩放三段角度使之和为360°。")
    scale = 360.0 / max(sum_ang, 1e-9)
    theta1 *= scale; theta2 *= scale; theta3 *= scale

# ---------- 运动律（简谐/余弦加速度） ----------
@st.cache_data
def build_motion(theta1, theta2, theta3, h1, n_pts):
    th = np.linspace(0.0, 360.0, n_pts)
    s = np.zeros_like(th)
    ds = np.zeros_like(th)    # ds/dθ (mm/deg)
    d2s = np.zeros_like(th)   # d²s/dθ² (mm/deg²)

    t0 = 0.0
    t1 = theta1
    t2 = theta1 + theta2

    # 升程：0 -> h1，s = h1 * 0.5 * (1 - cos(pi * tau))
    idx = (th >= t0) & (th <= t1)
    if np.any(idx):
        tau = (th[idx] - t0) / max(theta1, 1e-9)
        s[idx] = h1 * 0.5 * (1 - np.cos(np.pi * tau))
        ds[idx] = h1 * 0.5 * (np.pi / max(theta1, 1e-9)) * np.sin(np.pi * tau)
        d2s[idx] = h1 * 0.5 * (np.pi / max(theta1, 1e-9))**2 * np.cos(np.pi * tau)

    # 回程：h1 -> 0，s = h1 * 0.5 * (1 + cos(pi * tau))
    idx = (th > t1) & (th <= t2)
    if np.any(idx):
        tau = (th[idx] - t1) / max(theta2, 1e-9)
        s[idx] = h1 * 0.5 * (1 + np.cos(np.pi * tau))
        ds[idx] = -h1 * 0.5 * (np.pi / max(theta2, 1e-9)) * np.sin(np.pi * tau)
        d2s[idx] = -h1 * 0.5 * (np.pi / max(theta2, 1e-9))**2 * np.cos(np.pi * tau)

    # 静止段：保持 s 已为 0（回程结束）
    s[-1] = s[0]
    ds[-1] = ds[0]
    d2s[-1] = d2s[0]

    return th, s, ds, d2s

th_deg, s_mm, ds_dth, d2s_dth2 = build_motion(theta1, theta2, theta3, h1, pts)

# ---------- 旋转方向与角度映射 ----------
angle_sign = -1 if rotation_dir == '逆时针' else 1
# th_rad_seq 为 cam-fixed 下的角度（弧度），包括旋转方向
th_rad_seq = angle_sign * np.radians(th_deg)

# ---------- 轮廓生成：两种方案 ----------
if follower_type.startswith("roller"):
    # 点/滚子（原始简单情形）
    r_theta = r0 + s_mm
    cam_x = r_theta * np.cos(th_rad_seq)
    cam_y = r_theta * np.sin(th_rad_seq)
    profile = np.stack([cam_x, cam_y], axis=1)
    profile_method = "roller"
else:
    # 平底推杆：使用直线族包络法
    # 每个角度的法向 n = [cos φ, sin φ], d = r0 + s(φ)
    phi = th_rad_seq.copy()  # φ array
    n_x = np.cos(phi)
    n_y = np.sin(phi)
    d = r0 + s_mm

    # 用 phi 作为自变量，数值微分 d' = dd/dφ
    # 注意 phi 的步长可能为负（若 angle_sign=-1），np.gradient 能处理非均匀步长
    d_phi = np.gradient(d, phi)

    # n' = derivative wrt φ: [-sinφ, cosφ]
    npx = -np.sin(phi)
    npy =  np.cos(phi)

    # 对每个 φ 解线性方程 [ [n_x, n_y], [npx, npy] ] @ x = [d, d_phi]
    profile = np.zeros((len(phi), 2))
    bad_count = 0
    for i in range(len(phi)):
        A = np.array([[n_x[i], n_y[i]], [npx[i], npy[i]]])
        b = np.array([d[i], d_phi[i]])
        # 确保矩阵良好（对角线情况 A 的行列式应为 1，理论上应非奇异）
        detA = np.linalg.det(A)
        if abs(detA) < 1e-9:
            # 奇异点（理论上不应发生），退回到极坐标近似 r = d
            profile[i, :] = np.array([d[i]*n_x[i], d[i]*n_y[i]])
            bad_count += 1
        else:
            x = np.linalg.solve(A, b)
            profile[i, :] = x

    profile_method = "flat"
    if bad_count > 0:
        st.info(f"在计算平底包络时发现 {bad_count} 个近奇异点，已用径向近似替代这些点。")

# ---------- 绘图 ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('凸轮轮廓（cam-fixed）与接触示意')
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(profile[:,0], profile[:,1], '-', linewidth=2, label=f'cam profile ({profile_method})')
    base = Circle((0,0), r0, fill=False, linestyle='--', label='base circle r0')
    ax.add_patch(base)

    # 角度查看滑块（带 key）
    ang = st.slider('查看角度 θ (deg)', 0.0, 360.0, 0.0, 0.5, key="view_angle")
    # 找到最接近的索引（处理环绕）
    idx = int(np.argmin(np.abs(((th_deg - ang + 180) % 360) - 180)))

    if 0 <= idx < len(th_deg):
        pt = profile[idx]
        ax.plot([pt[0]],[pt[1]], 'ro', label='contact point')
        ax.plot([0, pt[0]],[0, pt[1]], '--', alpha=0.6)
        # 标注法向或推杆面（根据类型）
        if profile_method == "roller":
            # 法向即径向
            norm_pt = np.linalg.norm(pt)
            if norm_pt > 1e-9:
                nvec = pt / norm_pt
                ax.arrow(pt[0], pt[1], 0.1 * nvec[0], 0.1 * nvec[1],
                         head_width=max(0.01*r0, 0.2), length_includes_head=True, color='orange')
        else:
            # 平底：绘制该角度的直线 (法向 = n(φ))
            phi_i = th_rad_seq[idx]
            nvec = np.array([np.cos(phi_i), np.sin(phi_i)])
            # 画直线 (放在图内的一段)
            dirv = np.array([-nvec[1], nvec[0]])  # 与法向垂直
            seg = pt[None,:] + np.linspace(-0.5*r0, 0.5*r0, 2)[:,None] * dirv[None,:]
            ax.plot(seg[:,0], seg[:,1], ':', alpha=0.9, label='follower flat face (approx)')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.grid(True)
    ax.legend(loc='upper right')
    st.pyplot(fig)

with col2:
    st.subheader('位移 / 角速度 / 角加速度（θ 为自变量）')
    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.plot(th_deg, s_mm, '-', label='s(θ) [mm]')
    ax2.plot(th_deg, ds_dth, '--', label='ds/dθ [mm/deg]')
    ax2.plot(th_deg, d2s_dth2, ':', label='d²s/dθ² [mm/deg²]')
    ax2.axvline(th_deg[idx], color='k', linewidth=0.8, alpha=0.6)
    ax2.set_xlabel('θ (deg)')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# ---------- 导出 ----------
st.markdown('---')
st.subheader('导出数据')
if profile is not None and len(profile) > 0:
    df_prof = pd.DataFrame(profile, columns=['x_cam_mm','y_cam_mm'])
    # 根据从动件类型使用不同文件名
    prof_name = 'cam_profile_flat.csv' if profile_method == 'flat' else 'cam_profile_roller.csv'
    motion_name = 'cam_motion.csv'
    df_motion = pd.DataFrame({
        'theta_deg': th_deg,
        's_mm': s_mm,
        'ds_dtheta_mm_per_deg': ds_dth,
        'd2s_dtheta2_mm_per_deg2': d2s_dth2
    })
    st.download_button('下载 凸轮轮廓 CSV', df_prof.to_csv(index=False).encode('utf-8'), prof_name, 'text/csv', key="dl_profile")
    st.download_button('下载 运动学 CSV', df_motion.to_csv(index=False).encode('utf-8'), motion_name, 'text/csv', key="dl_motion")
