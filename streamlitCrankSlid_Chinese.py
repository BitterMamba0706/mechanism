# streamlit_slider_crank_fixed_cn.py
# 修正版：滑块-曲柄机构运动分析，支持生成 GIF 动画并下载
import io
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import streamlit as st
from font_config import set_chinese_font

# 设置中文字体
set_chinese_font()

# 设置中文字体，避免负号显示错误
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def calculate_kinematics(l1, l2, w1, theta_start_deg=0.0, theta_end_deg=360.0, N=361):
    theta1_deg = np.linspace(theta_start_deg, theta_end_deg, int(N))
    theta1 = np.deg2rad(theta1_deg)

    sin1, cos1 = np.sin(theta1), np.cos(theta1)

    sin2 = -(l1 * sin1) / l2
    sin2 = np.clip(sin2, -1.0, 1.0)
    theta2 = np.arcsin(sin2)
    cos2 = np.sqrt(1 - sin2**2)

    s3 = l1 * cos1 + l2 * cos2

    eps = 1e-12
    denom = (l2 * cos2)
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)

    omega2 = - (l1 * cos1 * w1) / denom
    v3 = - l1 * sin1 * w1 - l2 * sin2 * omega2

    alpha2 = (l1 * sin1 * w1**2 + l2 * sin2 * omega2**2) / denom
    a3 = - l1 * cos1 * w1**2 - l2 * cos2 * omega2**2 - l2 * sin2 * alpha2

    return theta1, theta2, s3, omega2, v3, alpha2, a3


def create_animation_bytes(l1, l2, theta_deg_array, fps=20, figsize=(6,5), bg_color=(255,255,255)):
    """生成 GIF 动画并返回字节流，适合 st.image 显示或下载"""
    theta1_rad = np.deg2rad(np.array(theta_deg_array, dtype=float))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color='gray', lw=1)
    ax.set_xlim(-l1 - l2 - 0.5, l1 + l2 + 0.5)
    ax.set_ylim(-l1 - l2 - 0.5, l1 + l2 + 0.5)
    ax.set_aspect('equal')
    ax.set_title("Slider-crank mechanism", fontsize=14, fontweight="bold")

    crank_line, = ax.plot([], [], 'o-', lw=3)
    rod_line, = ax.plot([], [], 'o-', lw=3)
    slider_line, = ax.plot([], [], 'ks-', lw=3)

    canvas = FigureCanvasAgg(fig)

    images = []
    for theta in theta1_rad:
        x_crank = l1 * np.cos(theta)
        y_crank = l1 * np.sin(theta)

        sin2 = -(l1 * np.sin(theta)) / l2
        sin2 = np.clip(sin2, -1.0, 1.0)
        cos2 = np.sqrt(max(0.0, 1 - sin2**2))
        s = l1 * np.cos(theta) + l2 * cos2

        crank_line.set_data([0, x_crank], [0, y_crank])
        rod_line.set_data([x_crank, s], [y_crank, 0])
        slider_line.set_data([s - 0.12, s + 0.12], [0, 0])

        canvas.draw()

        w, h = canvas.get_width_height()

        try:
            buf = canvas.tostring_rgb()
            img = Image.frombytes("RGB", (w, h), buf)
        except Exception:
            buf = canvas.buffer_rgba()
            arr = np.frombuffer(buf, dtype=np.uint8)
            arr = arr.reshape((h, w, 4))
            img = Image.fromarray(arr[..., :3])

        if img.mode != 'RGB':
            img = img.convert('RGB')
        bg = Image.new('RGB', img.size, bg_color)
        bg.paste(img)
        images.append(bg)

    if len(images) == 0:
        plt.close(fig)
        raise RuntimeError("未生成任何帧，请检查输入参数。")

    duration = int(1000 / fps)
    buf_io = io.BytesIO()
    images[0].save(
        buf_io,
        format='GIF',
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    buf_io.seek(0)

    plt.close(fig)
    return buf_io


st.set_page_config(page_title='滑块-曲柄运动分析', layout='wide')
st.title('滑块-曲柄机构运动分析 (Streamlit)')
st.markdown('计算位移、速度、加速度曲线，并生成 GIF 动画用于预览和下载。')

with st.sidebar.form('params'):
    st.header('输入参数设置')
    l1 = st.number_input('曲柄长度 l1', value=2.0, min_value=0.01, format='%.4f')
    l2 = st.number_input('连杆长度 l2', value=5.0, min_value=0.01, format='%.4f')
    w1 = st.number_input('曲柄角速度 ω1 (弧度/秒)', value=1.0, format='%.4f')
    theta_start = st.number_input('θ1 起始角度 (度)', value=0.0, format='%.2f')
    theta_end = st.number_input('θ1 结束角度 (度)', value=360.0, format='%.2f')
    N = st.number_input('计算步数 N', value=361, min_value=2, step=1)
    fps = st.slider('动画帧率 FPS', min_value=5, max_value=60, value=20)
    figsize_w = st.number_input('图像宽度 (英寸)', value=6.0, min_value=1.0, format='%.2f')
    figsize_h = st.number_input('图像高度 (英寸)', value=5.0, min_value=1.0, format='%.2f')
    submit = st.form_submit_button('开始计算并生成动画')

if submit:
    if theta_start > theta_end:
        st.error('起始角度必须小于或等于结束角度。')
    else:
        try:
            theta1, theta2, s3, omega2, v3, alpha2, a3 = calculate_kinematics(
                l1, l2, w1, theta_start_deg=theta_start, theta_end_deg=theta_end, N=int(N)
            )
            theta_deg = np.degrees(theta1)

            fig, axs = plt.subplots(3, 1, figsize=(figsize_w, figsize_h), constrained_layout=True)

            axs[0].plot(theta_deg, np.degrees(theta2), label='connecting rod angle θ2 (degree)')
            axs[0].plot(theta_deg, s3, '--', label='slider displacement s3')
            axs[0].set_title('displacement'); axs[0].legend(); axs[0].grid(True)

            axs[1].plot(theta_deg, omega2, label='connecting rod angular velocity ω2 (Radians per second)')
            axs[1].plot(theta_deg, v3, '--', label='slider speed v3')
            axs[1].set_title('speed'); axs[1].legend(); axs[1].grid(True)

            axs[2].plot(theta_deg, alpha2, label='Connecting rod angular acceleration α2')
            axs[2].plot(theta_deg, a3, '--', label='slider acceleration a3')
            axs[2].set_title('acceleration'); axs[2].legend(); axs[2].grid(True)

            st.pyplot(fig)

            with st.spinner('正在逐帧生成 GIF 动画，请稍候...'):
                theta_deg_array = np.linspace(theta_start, theta_end, int(N))
                gif_buf = create_animation_bytes(l1, l2, theta_deg_array, fps=fps, figsize=(figsize_w, figsize_h))

            st.success('动画已生成')

            # 显示 GIF 动画
            st.image(gif_buf.getvalue(), use_container_width=True)

            # 提供下载
            st.download_button(label='下载 GIF 动画', data=gif_buf.getvalue(),
                               file_name='slider_crank.gif', mime='image/gif')

        except Exception as e:
            st.exception(e)
else:
    st.info('请在左侧输入参数后，点击“开始计算并生成动画”。')
