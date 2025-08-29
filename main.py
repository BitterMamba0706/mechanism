# main.py
import streamlit as st

# 设置页面配置
st.set_page_config(page_title="GrayMario工作室", layout="centered")

# 导航
pg = st.navigation([
    st.Page("cover.py", title="导览", icon="🏠"),
    st.Page("AboutMe.py", title="关于作者", icon="🧩"),         
    st.Page("streamlitFourBar.py", title="四杆机构", icon="🔗"),   
    st.Page("streamlitCrankSlid_Chinese.py", title="曲柄滑块机构", icon="🔧"),  
    st.Page("Cam1.py", title="偏置直动滚子推杆盘形凸轮机构", icon="📐"),  
    st.Page("Cam2.py", title="对心直动平底/滚子推杆盘形凸轮机构", icon="📏"),  
    st.Page("Cam3.py", title="摆动滚子推杆盘形凸轮机构", icon="🌀"),  
    st.Page("Involute.py", title="渐开线曲线绘制器与渐开线函数表", icon="📊"),  
    st.Page("Gear.py", title="完整渐开线直齿圆柱齿轮绘制器（含整圈啮合线）", icon="⚙️"),  
    st.Page("GearCut.py", title="齿条插刀切制齿轮（动态仿真）", icon="🎞️"),  
])

pg.run()