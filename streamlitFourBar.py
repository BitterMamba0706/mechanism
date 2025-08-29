# streamlit_fourbar.py
import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from io import BytesIO
from PIL import Image, ImageDraw
import base64
import time
import matplotlib
from matplotlib import font_manager
from PIL import ImageFont
import os, base64
from matplotlib import font_manager as fm
from font_config import set_chinese_font

# 设置中文字体
set_chinese_font()

# ==================================================

# ---------------- 数学工具 ---------------- #
def circle_intersections(B, r0, D, r1):
    x0, y0 = B
    x1, y1 = D
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d < 1e-12 or d > (r0 + r1) or d < abs(r0 - r1):
        return []
    a = (r0**2 - r1**2 + d**2) / (2*d)
    h2 = r0**2 - a**2
    if h2 < -1e-12:
        return []
    h = math.sqrt(max(0, h2))
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * (h / d)
    ry = dx * (h / d)
    if abs(h) < 1e-12:
        return [np.array([xm, ym])]
    else:
        return [np.array([xm + rx, ym + ry]), np.array([xm - rx, ym - ry])]

def angle_of_vector(v):
    return math.atan2(v[1], v[0])

@dataclass
class FourBarParams:
    l1: float
    l2: float
    l3: float
    l4: float

class FourBarSolver:
    def __init__(self, params: FourBarParams):
        self.p = params
        self.A = np.array([0.0, 0.0])
        self.D = np.array([self.p.l4, 0.0])
        self.prev_C = None

    def solve_geometry(self, theta1_rad):
        B = self.A + self.p.l1 * np.array([math.cos(theta1_rad), math.sin(theta1_rad)])
        candidates = circle_intersections(B, self.p.l2, self.D, self.p.l3)
        if not candidates:
            return None
        if self.prev_C is not None and len(candidates) == 2:
            C = min(candidates, key=lambda c: np.linalg.norm(c - self.prev_C))
        else:
            C = sorted(candidates, key=lambda c: c[1])[-1]
        self.prev_C = C
        theta2 = angle_of_vector(C - B)
        theta3 = angle_of_vector(self.D - C)
        return B, C, theta2, theta3

    @staticmethod
    def _ui_vec(theta):
        return np.array([-math.sin(theta), math.cos(theta)])

    @staticmethod
    def _vi_vec(theta):
        return np.array([math.cos(theta), math.sin(theta)])

    def solve_vel_acc(self, theta1, theta2, theta3, w1, a1=0.0):
        u1 = self._ui_vec(theta1)
        u2 = self._ui_vec(theta2)
        u3 = self._ui_vec(theta3)
        v1 = self._vi_vec(theta1)
        v2 = self._vi_vec(theta2)
        v3 = self._vi_vec(theta3)
        A_mat = np.column_stack((self.p.l2 * u2, -self.p.l3 * u3))
        rhs = - self.p.l1 * w1 * u1
        det = np.linalg.det(A_mat)
        if abs(det) < 1e-10:
            return None
        w2, w3 = np.linalg.solve(A_mat, rhs)
        rhs_acc = (self.p.l1 * (w1**2 * v1 + a1 * u1)
                   + self.p.l2 * (w2**2) * v2 - self.p.l3 * (w3**2) * v3)
        a2, a3 = np.linalg.solve(A_mat, rhs_acc)
        return w2, w3, a2, a3

# ---------------- Streamlit 页面 ---------------- #
st.set_page_config(page_title='四杆机构运动学可视化', layout='wide')
st.title("四杆机构运动学可视化 (Streamlit)")

# 参数输入
st.sidebar.header("参数设置")
l1 = st.sidebar.number_input("l1 (长度)", value=100.0, min_value=0.1)
l2 = st.sidebar.number_input("l2 (长度)", value=200.0, min_value=0.1)
l3 = st.sidebar.number_input("l3 (长度)", value=250.0, min_value=0.1)
l4 = st.sidebar.number_input("l4 (机架长度)", value=300.0, min_value=0.1)

theta1_start = st.sidebar.number_input("theta1 起点 (°)", value=0.0)
theta1_end   = st.sidebar.number_input("theta1 终点 (°)", value=360.0)
N = st.sidebar.slider("步数 N", 10, 1000, 360)

w1 = st.sidebar.number_input("输入角速度 w1 (rad/s)", value=5.0)
a1 = st.sidebar.number_input("输入角加速度 a1 (rad/s²)", value=0.0)

# 用 session_state 保存计算结果，避免重复计算
if "computed" not in st.session_state:
    st.session_state.computed = False

if st.sidebar.button("计算并绘图"):
    p = FourBarParams(l1, l2, l3, l4)
    solver = FourBarSolver(p)

    thetas1 = np.linspace(math.radians(theta1_start), math.radians(theta1_end), N)
    theta2 = np.full(N, np.nan)
    theta3 = np.full(N, np.nan)
    w2 = np.full(N, np.nan)
    w3 = np.full(N, np.nan)
    a2 = np.full(N, np.nan)
    a3 = np.full(N, np.nan)

    B_list, C_list = [], []
    for i, t1 in enumerate(thetas1):
        geo = solver.solve_geometry(t1)
        if geo is None:
            B_list.append(None); C_list.append(None)
            continue
        B, C, t2, t3 = geo
        theta2[i], theta3[i] = t2, t3
        B_list.append(B); C_list.append(C)
        velacc = solver.solve_vel_acc(t1, t2, t3, w1=w1, a1=a1)
        if velacc is not None:
            w2[i], w3[i], a2[i], a3[i] = velacc

    theta1_deg = np.degrees(thetas1)

    # 保存到 session_state，供其他按钮使用（避免重新计算）
    st.session_state.p = p
    st.session_state.thetas1 = thetas1
    st.session_state.theta1_deg = theta1_deg
    st.session_state.theta2 = theta2
    st.session_state.theta3 = theta3
    st.session_state.w2 = w2
    st.session_state.w3 = w3
    st.session_state.a2 = a2
    st.session_state.a3 = a3
    st.session_state.B_list = B_list
    st.session_state.C_list = C_list
    st.session_state.computed = True

# 只有在已经计算过的情况下才绘图 / 生成 GIF / 保存 CSV
if st.session_state.computed:
    p = st.session_state.p
    theta1_deg = st.session_state.theta1_deg
    theta2 = st.session_state.theta2
    theta3 = st.session_state.theta3
    w2 = st.session_state.w2
    w3 = st.session_state.w3
    a2 = st.session_state.a2
    a3 = st.session_state.a3
    B_list = st.session_state.B_list
    C_list = st.session_state.C_list
    A = np.array([0, 0])
    D = np.array([p.l4, 0])

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ["Microsoft YaHei"]   # 例如 "Microsoft YaHei" 或 直接 ttf 路径
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制角位移
    fig1, ax1 = plt.subplots()
    ax1.plot(theta1_deg, np.degrees(theta2), label="theta2 (deg)")
    ax1.plot(theta1_deg, np.degrees(theta3), label="theta3 (deg)")
    ax1.set_xlabel("theta1 (deg)")
    ax1.set_ylabel("angular displacement (deg)")
    ax1.legend(); ax1.grid(True)
    st.pyplot(fig1)

    # 绘制角速度
    fig2, ax2 = plt.subplots()
    ax2.plot(theta1_deg, w2, label="w2 (rad/s)")
    ax2.plot(theta1_deg, w3, label="w3 (rad/s)")
    ax2.set_xlabel("theta1 (deg)")
    ax2.set_ylabel("angular velocity (rad/s)")
    ax2.legend(); ax2.grid(True)
    st.pyplot(fig2)

    # 绘制角加速度
    fig3, ax3 = plt.subplots()
    ax3.plot(theta1_deg, a2, label="a2 (rad/s²)")
    ax3.plot(theta1_deg, a3, label="a3 (rad/s²)")
    ax3.set_xlabel("theta1 (deg)")
    ax3.set_ylabel("angular acceleration (rad/s²)")
    ax3.legend(); ax3.grid(True)
    st.pyplot(fig3)

    # 机构示意图（取第一帧）
    fig4, ax4 = plt.subplots()
    B = B_list[0]; C = C_list[0]
    ax4.plot([A[0], D[0]], [A[1], D[1]], "k-", lw=2)
    if B is not None and C is not None:
        ax4.plot([A[0], B[0]], [A[1], B[1]], "b-")
        ax4.plot([B[0], C[0]], [B[1], C[1]], "g-")
        ax4.plot([C[0], D[0]], [C[1], D[1]], "r-")
        ax4.plot([A[0], B[0], C[0], D[0]], [A[1], B[1], C[1], D[1]], "ko")
    ax4.set_aspect("equal")
    ax4.set_title("Initial position of the four-bar mechanism")
    st.pyplot(fig4)


    # ---------- 改进版：按全局 bounding box 自适应画布并生成 GIF ----------
    if st.button("生成并播放 GIF 动画（自适应尺寸）"):
        # 收集所有有效点，计算 bounding box
        xs, ys = [], []
        for i in range(len(B_list)):
            Bi = B_list[i]; Ci = C_list[i]
            if Bi is not None and Ci is not None:
                pts = [A, Bi, Ci, D]
                for p in pts:
                    xs.append(p[0]); ys.append(p[1])
        if not xs:
            st.warning("所有帧都无解，无法生成完整动画（检查连杆长度与运动范围）。")
        else:
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            width_units = maxx - minx if maxx - minx > 1e-9 else 1.0
            height_units = maxy - miny if maxy - miny > 1e-9 else 1.0
            margin_units = 0.12 * max(width_units, height_units)  # 留 12% 空白边距

            # 目标最大像素（你可以改这个值）
            target_max_pixels = 640
            # 计算缩放比例（units -> pixels）
            scale = min((target_max_pixels - 40) / (width_units + 2 * margin_units),
                        (target_max_pixels - 40) / (height_units + 2 * margin_units))
            if scale <= 0:
                scale = 1.0

            left_pad = 20
            bottom_pad = 20
            canvas_w = int((width_units + 2 * margin_units) * scale) + left_pad * 2
            canvas_h = int((height_units + 2 * margin_units) * scale) + bottom_pad * 2

            # 映射函数：物理坐标 -> 像素坐标
            def phys_to_px(p):
                x, y = p[0], p[1]
                px = left_pad + (x - minx + margin_units) * scale
                py = canvas_h - (bottom_pad + (y - miny + margin_units) * scale)
                return (px, py)

            # 生成帧
            frames = []
            for i in range(len(B_list)):
                img = Image.new("RGBA", (canvas_w, canvas_h), "white")
                draw = ImageDraw.Draw(img)
                Bi = B_list[i]; Ci = C_list[i]
                # 先画基座线 AD
                draw.line([phys_to_px(A), phys_to_px(D)], fill="black", width=3)
                if Bi is not None and Ci is not None:
                    pts_phys = [A, Bi, Ci, D]
                    pts_px = [phys_to_px(p) for p in pts_phys]
                    draw.line([pts_px[0], pts_px[1]], fill="blue", width=4)   # AB
                    draw.line([pts_px[1], pts_px[2]], fill="green", width=4)  # BC
                    draw.line([pts_px[2], pts_px[3]], fill="red", width=4)    # CD
                    # 关节
                    for pt in pts_px:
                        draw.ellipse([pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5], fill="black")
                else:
                    # 如果当前帧无解，画一条浅灰的横线表示“无解帧”
                    # 也可只显示 AD 基座
                    draw.text((10, 10), "frame no solution", fill="gray")
                frames.append(img.convert("P", palette=Image.ADAPTIVE))

            # 保存 GIF 到 bytes
            gif_bytes = BytesIO()
            frames[0].save(
                gif_bytes,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
                optimize=False,
                disposal=2
            )
            gif_bytes.seek(0)

            # 嵌入并在浏览器播放（同时提供下载）
            b64 = base64.b64encode(gif_bytes.read()).decode("ascii")
            html = f'<div style="text-align:center;"><img src="data:image/gif;base64,{b64}" alt="fourbar" style="max-width:100%; height:auto; border:1px solid #ddd;"></div>'
            st.markdown(html, unsafe_allow_html=True)

            gif_bytes.seek(0)
            st.download_button("下载 GIF 动画", data=gif_bytes.getvalue(), file_name="fourbar_adapt.gif", mime="image/gif")

    # 备用：用占位符逐帧刷新（改进版：按全局 bounding box 自适应画布并逐帧显示）
    if st.button("在页面逐帧显示动画（占位符刷新）"):
        placeholder = st.empty()

        # 先收集所有帧中的点，计算全局 bounding box（保证不被裁剪）
        xs, ys = [], []
        for i in range(len(B_list)):
            Bi = B_list[i]; Ci = C_list[i]
            if Bi is not None and Ci is not None:
                for pnt in (A, Bi, Ci, D):
                    xs.append(pnt[0]); ys.append(pnt[1])
        if not xs:
            st.warning("所有帧都无解，无法逐帧显示动画（请检查连杆长度与运动范围）。")
        else:
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            width_units = maxx - minx if (maxx - minx) > 1e-9 else 1.0
            height_units = maxy - miny if (maxy - miny) > 1e-9 else 1.0

            # 边距（单位坐标）
            margin_units = 0.12 * max(width_units, height_units)

            # 设定目标最大像素尺寸（若想更大/更小可修改）
            target_max_pixels = 640

            # 计算从“物理单位”到像素的缩放比例
            scale = min(
                (target_max_pixels - 40) / (width_units + 2 * margin_units),
                (target_max_pixels - 40) / (height_units + 2 * margin_units),
            )
            if scale <= 0:
                scale = 1.0

            left_pad = 20
            bottom_pad = 20
            canvas_w = int((width_units + 2 * margin_units) * scale) + left_pad * 2
            canvas_h = int((height_units + 2 * margin_units) * scale) + bottom_pad * 2

            # 坐标转换：物理坐标 -> 像素坐标（注意 y 轴翻转以符合图像坐标）
            def phys_to_px(p):
                x, y = p[0], p[1]
                px = left_pad + (x - minx + margin_units) * scale
                py = canvas_h - (bottom_pad + (y - miny + margin_units) * scale)
                return (px, py)

            # 逐帧绘制并刷新占位符
            for i in range(len(B_list)):
                img = Image.new("RGB", (canvas_w, canvas_h), "white")
                draw = ImageDraw.Draw(img)

                Bi = B_list[i]; Ci = C_list[i]

                # 画机架 AD（基座）始终可见
                draw.line([phys_to_px(A), phys_to_px(D)], fill="black", width=3)

                if Bi is not None and Ci is not None:
                    pts_phys = [A, Bi, Ci, D]
                    pts_px = [phys_to_px(pnt) for pnt in pts_phys]
                    # 绘制连杆
                    draw.line([pts_px[0], pts_px[1]], fill="blue", width=4)   # AB
                    draw.line([pts_px[1], pts_px[2]], fill="green", width=4)  # BC
                    draw.line([pts_px[2], pts_px[3]], fill="red", width=4)    # CD
                    # 绘制关节点
                    for pt in pts_px:
                        draw.ellipse([pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5], fill="black")
                else:
                    # 无解帧：在画面上显示提示（同时保持 AD 可见）
                    draw.text((10, 10), "frame no solution", fill="gray")

                # 显示当前帧（use_column_width 可选，若想按像素显示可去掉）
                placeholder.image(img, use_container_width=False)

                # 控制帧率（可按需调整）
                time.sleep(0.03)


    # ---------------- 保存数据为 CSV ---------------- #
    if st.button("保存数据为 CSV 文件"):
        data_dict = {
            "theta1_deg": theta1_deg,
            "theta2_deg": np.degrees(theta2),
            "theta3_deg": np.degrees(theta3),
            "w2_rad_s": w2,
            "w3_rad_s": w3,
            "a2_rad_s2": a2,
            "a3_rad_s2": a3,
        }
        df = pd.DataFrame(data_dict)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("下载 CSV 文件", data=csv_bytes, file_name="fourbar_data.csv", mime="text/csv")
else:
    st.info("请在左侧设置参数并点击 “计算并绘图” 来生成数据和图像。")
