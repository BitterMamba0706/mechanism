# cover.py
import streamlit as st
from PIL import Image

# 设置页面配置
st.set_page_config(page_title="GrayMario工作室 - 机械机构分析", layout="centered")

# 加载机械相关背景图（可以替换为本地图片路径或网络图片）
st.image("assets/gears.gif", 
         caption="机械齿轮运转", use_container_width=True)

# 封面标题
st.markdown("<h1 style='text-align: center; color: steelblue;'>GrayMario工作室</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>机械机构运动分析平台</h3>", unsafe_allow_html=True)

# 简介
st.write("""
欢迎来到 **GrayMario工作室** 🚀  
本平台致力于提供 **四杆机构、曲柄滑块、盘形凸轮等典型机构** 的运动可视化与分析工具。  
请选择左侧导航进入对应模块开始探索！
""")

# 底部版权
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2025 GrayMario工作室 | 机械机构运动分析平台</p>", unsafe_allow_html=True)
