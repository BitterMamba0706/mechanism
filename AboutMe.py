# streamlit_intro.py 
# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import os
import base64
import html

st.set_page_config(page_title="关于GrayMario", page_icon="🤖", layout="wide")

# --------------------------
# 背景媒体路径 & 存在性检查
# --------------------------
mp4_path = "assets/Japan.mp4"
webm_path = "assets/Japan.webm"
gif_path = "assets/Japan.gif"   # 页面背景 gif（优先用于 gif 回退）
dog_gif_path = "assets/dog.gif" # 右下角柴犬 gif（用户会提供）

has_mp4 = os.path.exists(mp4_path)
has_webm = os.path.exists(webm_path)
has_gif = os.path.exists(gif_path)
has_dog = os.path.exists(dog_gif_path)

# 如果存在 gif/dog gif，则把它们读成 base64（内嵌在 HTML/CSS 中，避免静态文件引用问题）
bg_gif_data = ""

# 在读取 bg_gif_data 后加入：直接注入 fixed 背景 div（更鲁棒）
if bg_gif_data:
    bg_div_html = f'''
    <div id="__bg_gif" style="
        position:fixed;
        inset:0;
        z-index:-99999;
        pointer-events:none;
        background: url('data:image/gif;base64,{bg_gif_data}') center center / cover no-repeat;
        filter: brightness(0.68) saturate(1.05);
    "></div>
    <div id="__bg_overlay" style="
        position:fixed;
        inset:0;
        z-index:-99998;
        pointer-events:none;
        background: linear-gradient(120deg, rgba(2,6,23,0.62) 0%, rgba(6,12,30,0.62) 55%, rgba(0,0,6,0.72) 100%);
    "></div>
    '''
    # height 必须是正数；1 足够小且不影响布局
    components.html(bg_div_html, height=1, scrolling=False)

dog_gif_data = ""
if has_gif:
    try:
        with open(gif_path, "rb") as f:
            bg_gif_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        bg_gif_data = ""
if has_dog:
    try:
        with open(dog_gif_path, "rb") as f:
            dog_gif_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        dog_gif_data = ""

# --------------------------
# 插入 <video> 背景（或回退到 gif / 渐变）
# --------------------------
if has_mp4 or has_webm:
    src = mp4_path if has_mp4 else webm_path
    video_html = f"""
    <div style="position:fixed; inset:0; z-index:-9999; pointer-events:none;">
      <video id="bg-video" autoplay muted loop playsinline style="width:100%; height:100%; object-fit:cover; display:block; filter:brightness(0.68) saturate(1.05);">
        <source src="{src}" type="{'video/mp4' if has_mp4 else 'video/webm'}">
        Your browser does not support the video tag.
      </video>
    </div>
    <div style="position:fixed; inset:0; background: linear-gradient(120deg, rgba(2,6,23,0.62) 0%, rgba(6,12,30,0.62) 55%, rgba(0,0,6,0.72) 100%); z-index:-9998; pointer-events:none;"></div>
    """
    components.html(video_html, height=8, scrolling=False)
    bg_style_variant = "video"
elif has_gif and bg_gif_data:
    bg_style_variant = "gif"
else:
    bg_style_variant = "none"

# --------------------------
# 全局 CSS（签名/卡片/柴犬等样式）
# 如果使用内嵌 GIF，采用 data URI，这样 Streamlit 可以正确加载背景
# --------------------------
if bg_style_variant == "gif" and bg_gif_data:
    BG_CSS = f"""
    background:
      linear-gradient(120deg, rgba(2,6,23,0.72) 0%, rgba(6,12,30,0.72) 55%, rgba(0,0,6,0.82) 100%),
      url("data:image/gif;base64,{bg_gif_data}") center center / cover no-repeat fixed;
    """
elif bg_style_variant == "none":
    BG_CSS = """
    background:
      radial-gradient(1200px 600px at 10% 10%, rgba(12,10,54,0.9), transparent 6%),
      linear-gradient(120deg, rgba(2,6,23,1) 0%, rgba(6,12,30,1) 55%, rgba(0,0,6,1) 100%);
    background-attachment: fixed;
    """
else:
    # video 使用半透明叠层
    BG_CSS = """
    background:
      linear-gradient(120deg, rgba(2,6,23,0.20) 0%, rgba(6,12,30,0.20) 55%, rgba(0,0,6,0.25) 100%);
    """

STYLES = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Permanent+Marker&family=Pacifico&display=swap');
:root{{ --glass-bg: rgba(255,255,255,0.03); --accent1: #00f5ff; --accent2: #9d00ff; --accent3: #ff7ad1; --glass-border: rgba(255,255,255,0.06); --card-shadow: 0 10px 50px rgba(0,0,0,0.65); --glass-radius: 14px; }}

/* 全局背景 & 字体 */
html, body, [data-testid="stAppViewContainer"] > .main {{
  min-height:100vh;
  {BG_CSS}
  color: #E6F0FF;
  font-family: Inter, "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing:antialiased;
}}

/* header / signature / cards / shiba 样式（保留你原来的风格） */
.header-neo{{ border-radius: 14px; padding: 18px; margin-bottom: 18px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: var(--card-shadow); border: 1px solid var(--glass-border); }}
.title-neon {{ font-size: 34px; font-weight: 800; letter-spacing: 0.6px; color: #E6F0FF; text-shadow: 0 0 18px rgba(0,245,255,0.12), 0 0 36px rgba(157,0,255,0.08), 0 6px 20px rgba(0,0,0,0.6); }}
.signature-wrap {{ margin-top: 8px; display:flex; align-items:center; gap:14px; }}
.signature {{ font-family: "Permanent Marker", "Pacifico", "Brush Script MT", cursive; font-size:58px; line-height:1; letter-spacing:1px; display:inline-block; color: transparent; background: linear-gradient(90deg, var(--accent1), var(--accent2), var(--accent3)); -webkit-background-clip: text; background-clip: text; transform: skew(-6deg) rotate(-2deg); text-shadow: 0 0 10px rgba(0,245,255,0.18), 0 0 22px rgba(157,0,255,0.12), 0 8px 26px rgba(0,0,0,0.6); animation: sig-reveal 1.6s forwards ease-out, signature-float 6s ease-in-out 2s infinite; }}
@keyframes sig-reveal {{ 0% {{ clip-path: inset(0 100% 0 0); filter: blur(2px); opacity:0; transform: translateY(6px) skew(-6deg) rotate(-2deg); }} 60% {{ clip-path: inset(0 5% 0 0); opacity:1; filter: blur(0.6px); }} 100% {{ clip-path: inset(0 0% 0 0); filter: blur(0); opacity:1; transform: translateY(0) skew(-6deg) rotate(-2deg); }} }}
@keyframes signature-float {{ 0% {{ transform: translateY(0) skew(-6deg) rotate(-1.5deg); background-position: 0% 50%; }} 50% {{ transform: translateY(-6px) skew(-6deg) rotate(-2.5deg); background-position: 50% 50%; }} 100% {{ transform: translateY(0) skew(-6deg) rotate(-1.5deg); background-position: 100% 50%; }} }}
.signature-pen {{ width:12px; height:12px; border-radius:50%; background: radial-gradient(circle at 30% 30%, #fff, #ffd1f8 30%, transparent 60%); box-shadow: 0 8px 28px rgba(157,0,255,0.35), 0 2px 6px rgba(0,0,0,0.45); position:absolute; top:48%; left:-6%; transform: translateY(-50%) scale(0.85); opacity:0; animation: pen-move 1.7s forwards 0.08s cubic-bezier(.2,.9,.3,1); }}
@keyframes pen-move {{ 0% {{ left: -12%; opacity: 0; transform: translateY(-50%) scale(0.6); }} 10% {{ opacity: 1; }} 60% {{ left: 92%; transform: translateY(-50%) scale(1.05); }} 100% {{ left: 130%; opacity: 0; transform: translateY(-50%) scale(0.9); }} }}
.card {{ background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01)); border: 1px solid var(--glass-border); border-radius: var(--glass-radius); padding: 14px; margin-bottom: 12px; box-shadow: var(--card-shadow); }}
.gif-grid{{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:10px; }}
.gif-grid img{{ border-radius:10px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 12px 30px rgba(0,0,0,0.6); height:140px; width:100%; object-fit:cover; transition: transform .35s ease, box-shadow .35s ease; }}
.gif-grid img:hover{{ transform: translateY(-6px) scale(1.03); box-shadow:0 22px 46px rgba(0,0,0,0.7); }}
.tag {{ display:inline-block; padding:6px 10px; margin:4px; border-radius:999px; font-size:13px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.03); }}
.footer {{ font-size:13px; color: #9FB7D9; opacity:0.9; padding-top:8px; }}

/* -----------------------------
   右下角柴犬交互样式（更生动）   ----------------------------- */
.shiba {{
  position: fixed;
  right: 18px;
  bottom: 18px;
  width:92px;
  height:92px;
  border-radius:50%;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:46px;
  background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 18px 40px rgba(0,0,0,0.6), 0 6px 20px rgba(157,0,255,0.08);
  cursor: pointer;
  z-index: 10001;
  user-select: none;
  transform-origin: center;
  transition: box-shadow .18s ease, transform .18s ease;
  overflow: visible;
  pointer-events: auto; /* 允许交互 */
}}
.shiba-img {{
  width:88px;
  height:88px;
  border-radius:50%;
  object-fit:cover;
  display:block;
  transform-origin: center bottom;
  will-change: transform;
  transition: transform 260ms cubic-bezier(.2,.9,.3,1);
  box-shadow: 0 8px 24px rgba(0,0,0,0.45);
}}

/* 轻微呼吸动画 */
@keyframes shiba-breathe {{
  0% {{ transform: translateY(0) scale(1); }}
  50% {{ transform: translateY(-3px) scale(1.02); }}
  100% {{ transform: translateY(0) scale(1); }}
}}
.shiba:not(.shiba-activated) .shiba-img {{
  animation: shiba-breathe 3.8s ease-in-out infinite;
}}

/* 点击激活的夸张弹跳 + 轻微旋转（保留原先设计） */
.shiba.shiba-activated {{
  transform: translateY(0) scale(1.06);
  box-shadow: 0 28px 60px rgba(0,0,0,0.7), 0 10px 32px rgba(157,0,255,0.12);
}}
@keyframes shiba-poke {{
  0% {{ transform: translateY(0) rotate(-1deg) scale(1); }}
  20% {{ transform: translateY(-22px) rotate(18deg) scale(1.12); }}
  45% {{ transform: translateY(-6px) rotate(-8deg) scale(1.04); }}
  70% {{ transform: translateY(-10px) rotate(8deg) scale(1.06); }}
  100% {{ transform: translateY(0) rotate(0deg) scale(1); }}
}}
.shiba.shiba-activated .shiba-img {{
  animation: shiba-poke 900ms cubic-bezier(.22,.9,.3,1);
}}

/* 气泡提示（点击时短暂出现） */
.shiba-bubble {{
  position: absolute;
  right: 108%;
  bottom: 20%;
  min-width:72px;
  padding:6px 10px;
  border-radius:12px;
  background: rgba(2,12,30,0.9);
  color:#BEEBFF;
  font-weight:700;
  font-size:13px;
  text-align:center;
  box-shadow: 0 8px 26px rgba(0,0,0,0.6);
  opacity:0;
  transform: translateY(6px) scale(0.92);
  pointer-events: none;
  transition: opacity .18s ease, transform .28s cubic-bezier(.2,.9,.3,1);
  z-index: 10002;
}}
.shiba.shiba-activated .shiba-bubble {{
  opacity:1;
  transform: translateY(-6px) scale(1);
}}

/* 小脚印（保留） */
.shiba::after{{ content: "🐾"; position: absolute; right: -8px; bottom: 10px; font-size:14px; opacity:0.9; transform-origin: center; animation: paw-wag 1.8s infinite ease-in-out; }}

/* 其它动画 */
@keyframes paw-wag {{ 0% {{ transform: rotate(-12deg) translateY(0); opacity:0.9; }} 50% {{ transform: rotate(8deg) translateY(-2px); opacity:1; }} 100% {{ transform: rotate(-12deg) translateY(0); opacity:0.9; }} }}

.tree-wrap {{ display:flex; align-items:center; justify-content:center; padding:8px 4px; }}
.tree-caption {{ color:#B8E6B0; font-size:13px; margin-top:8px; text-align:center; opacity:0.9; }}
@media (max-width: 760px) {{ .tree-svg {{ width:220px; height:220px; }} }}

/* --- Live2D widget 提示样式覆盖，确保在页面左下角并可交互 --- */
.live2d-widget-container {{
  z-index: 10010 !important;
  pointer-events: auto !important;
}}
.live2d-widget-canvas {{
  pointer-events: auto !important;
}}

/* 全页粒子画布（canvas） - 我插入的样式，z-index 低于 Live2D 和 shiba，且不阻挡交互 */
#particle-canvas {{
  position: fixed;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  z-index: 10000;
  pointer-events: none; /* 不阻挡页面交互 */
}}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# --------------------------
# Canvas + JS: 鼠标拖尾 + 粒子连线 + 烟花
# 我把全部逻辑写在一个脚本内，直接注入到主页面
# --------------------------
particle_script = """
<canvas id="particle-canvas"></canvas>
<script>
(function(){
  const canvas = document.getElementById('particle-canvas');
  const ctx = canvas.getContext('2d');
  let W = canvas.width = window.innerWidth;
  let H = canvas.height = window.innerHeight;

  // 鼠标跟踪（window 监听，保持 pointer-events:none 时也能接收）
  const mouse = { x: W/2, y: H/2, down: false };

  window.addEventListener('mousemove', (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
  window.addEventListener('resize', () => { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; initNetwork(); });
  window.addEventListener('mousedown', (e) => { mouse.down = true; spawnFirework(e.clientX, e.clientY); });
  window.addEventListener('mouseup', () => { mouse.down = false; });

  /* -----------------------
     拖尾粒子（流线）
     ----------------------- */
  function TrailParticle(x,y,r,color){
    this.x = x; this.y = y; this.r = r; this.color = color;
    this.radians = Math.random()*Math.PI*2;
    this.velocity = 0.06 + Math.random()*0.04;
    this.distance = 20 + Math.random()*80;
    this.lastMouse = {x: x, y: y};
  }
  TrailParticle.prototype.update = function(){
    this.radians += this.velocity;
    // 平滑跟随
    this.lastMouse.x += (mouse.x - this.lastMouse.x) * 0.12;
    this.lastMouse.y += (mouse.y - this.lastMouse.y) * 0.12;
    const lx = this.lastMouse.x + Math.cos(this.radians) * this.distance;
    const ly = this.lastMouse.y + Math.sin(this.radians) * this.distance;
    // 轻微移动当前点（让轨迹更流畅）
    this.x += (lx - this.x) * 0.28;
    this.y += (ly - this.y) * 0.28;
  };
  TrailParticle.prototype.draw = function(){
    ctx.beginPath();
    ctx.strokeStyle = this.color;
    ctx.lineWidth = this.r;
    ctx.moveTo(this.x, this.y);
    // 绘制一个短线段作为拖尾的一部分（通过历史记录也可以更长）
    const dx = Math.cos(this.radians + Math.PI) * this.r * 1.2;
    const dy = Math.sin(this.radians + Math.PI) * this.r * 1.2;
    ctx.lineTo(this.x+dx, this.y+dy);
    ctx.stroke();
    ctx.closePath();
  };

  const trailColors = ['#00bdff','#4d39ce','#088eff','#ff7ad1','#ffd166'];
  let trailParticles = [];
  function initTrail(n=45){
    trailParticles = [];
    for(let i=0;i<n;i++){
      const p = new TrailParticle(W/2, H/2, Math.random()*2+1, trailColors[Math.floor(Math.random()*trailColors.length)]);
      trailParticles.push(p);
    }
  }

  /* -----------------------
     粒子连线网络（基于第二段代码）
     ----------------------- */
  let network = [];
  function initNetwork(n=60){
    network = [];
    for(let i=0;i<n;i++){
      const px = Math.random()*W;
      const py = Math.random()*H;
      const xa = (Math.random()*2-1) * 0.6;
      const ya = (Math.random()*2-1) * 0.6;
      network.push({x:px, y:py, xa:xa, ya:ya, max: 6000});
    }
  }

  /* -----------------------
     烟花（点击生成）
     ----------------------- */
  let fireworks = [];
  function Firework(x,y,color){
    this.x = x; this.y = y; this.color = color;
    this.particles = [];
    const count = 30 + Math.floor(Math.random()*30);
    for(let i=0;i<count;i++){
      const angle = Math.random()*Math.PI*2;
      const speed = 1.5 + Math.random()*4;
      const vx = Math.cos(angle) * speed;
      const vy = Math.sin(angle) * speed;
      this.particles.push({
        x: x, y: y, vx: vx, vy: vy,
        life: 60 + Math.floor(Math.random()*50),
        age: 0,
        r: 1 + Math.random()*2,
        color: color
      });
    }
  }
  function spawnFirework(x,y){
    const palette = ['#FF4D4D','#FFAE42','#FFD166','#7BD389','#9D5CFF','#00E5FF','#FF7AD1'];
    const color = palette[Math.floor(Math.random()*palette.length)];
    fireworks.push(new Firework(x,y,color));
  }

  /* -----------------------
     主动画循环
     ----------------------- */
  function animate(){
    requestAnimationFrame(animate);
    // 轻微擦除（透明填充以产生拖尾残影）
    ctx.fillStyle = 'rgba(2,6,23,0.20)';
    ctx.fillRect(0,0,W,H);

    // 网络粒子移动与连线
    for(let i=0;i<network.length;i++){
      const p = network[i];
      p.x += p.xa;
      p.y += p.ya;
      if(p.x > W || p.x < 0) p.xa *= -1;
      if(p.y > H || p.y < 0) p.ya *= -1;
      // 绘制点
      ctx.fillStyle = 'rgba(180,200,255,0.06)';
      ctx.fillRect(p.x-0.5, p.y-0.5, 1, 1);
      // 连线到其它粒子
      for(let j=i+1;j<network.length;j++){
        const q = network[j];
        const dx = p.x - q.x;
        const dy = p.y - q.y;
        const dist2 = dx*dx + dy*dy;
        if(dist2 < 5000){
          const t = 1 - dist2/5000;
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(120,160,255,' + (0.06 * t + 0.02) + ')';
          ctx.lineWidth = 1 * (0.4 * t + 0.1);
          ctx.moveTo(p.x,p.y);
          ctx.lineTo(q.x,q.y);
          ctx.stroke();
          ctx.closePath();
        }
      }
      // 与鼠标连线（增强互动感）
      if(mouse.x !== null && mouse.y !== null){
        const dxm = p.x - mouse.x, dym = p.y - mouse.y;
        const dm2 = dxm*dxm + dym*dym;
        if(dm2 < 90000){
          const tt = 1 - dm2/90000;
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(150,200,255,' + (0.06 * tt + 0.01) + ')';
          ctx.lineWidth = 1 * (0.3 * tt + 0.06);
          ctx.moveTo(p.x,p.y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.stroke();
          ctx.closePath();
        }
      }
    }

    // 拖尾粒子更新与绘制（在网络之上）
    for(let i=0;i<trailParticles.length;i++){
      const tp = trailParticles[i];
      tp.update();
      tp.draw();
    }

    // 烟花更新
    for(let k = fireworks.length-1; k>=0; k--){
      const fw = fireworks[k];
      let alive = false;
      for(let i=0;i<fw.particles.length;i++){
        const fp = fw.particles[i];
        // physics
        fp.vy += 0.06; // gravity
        fp.x += fp.vx;
        fp.y += fp.vy;
        fp.vx *= 0.995;
        fp.vy *= 0.995;
        fp.age++;
        // fade
        const lifeRatio = 1 - fp.age / fp.life;
        if(lifeRatio > 0){
          alive = true;
          ctx.beginPath();
          ctx.fillStyle = fp.color;
          ctx.globalAlpha = Math.max(0, lifeRatio);
          ctx.fillRect(fp.x - fp.r, fp.y - fp.r, fp.r*2, fp.r*2);
          ctx.globalAlpha = 1;
          ctx.closePath();
        }
      }
      if(!alive){
        fireworks.splice(k,1);
      }
    }
  }

  // 初始化
  initTrail(45);
  initNetwork(70);
  animate();

  // 保障当页面切到后台或重新加载时不出问题
  window.addEventListener('blur', ()=>{ /* no-op */ });

  // 小性能优化：如果需要可在高 DPI 下缩放 canvas（这里暂且不做）
})();
</script>
"""

components.html(particle_script, height=0, scrolling=False)


# --------------------------
# 固定作者信息（在代码中直接填写）
# --------------------------
name = "Gray Mario"
age = 21
undergrad = "NJFU"
postgrad = "DHU"
hobbies = ["Study", "Travel", "BasketBall"]
twitter = "https://x.com/MG_forever30"
instagram = "https://instagram.com/bittermamba30"
gmail = "mgray0706@gmail.com"
bio_short = "努力学习打工赚钱梦想环游世界"
avatar_path = "assets/LM10.jpg"
gif_paths = ["assets/Earth.gif"]
tree_gif_path = "assets/Tree.gif"

# --------------------------
# Header（包含酷炫签名）
# --------------------------
with st.container():
    st.markdown('<div class="header-neo">', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f'<div class="title-neon">👾 {name} </div>', unsafe_allow_html=True)
        sig_html = f'''
        <div class="signature-wrap">
          <div style="position:relative; display:inline-block;">
            <div class="signature">{name}</div>
            <div class="signature-pen"></div>
          </div>
          <div style="color:#B8D8FF;padding-top:6px;font-size:13px;">{bio_short}</div>
        </div>
        '''
        st.markdown(sig_html, unsafe_allow_html=True)
    with col2:
        try:
            avatar = Image.open(avatar_path)
            st.image(avatar, width=112)
        except Exception:
            st.markdown('<div class="tag">未上传头像</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# 三列布局（主内容）
# --------------------------
left, middle, right = st.columns([1.1, 2.2, 1.1])

# 左：关于 + Tree.gif 动画替换原 SVG（显示 GIF 并显示新的说明文字）
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("关于我")
    st.write(f"**{name}**  ·  {age} 岁")
    st.write(f"🎓 {undergrad} → {postgrad}")
    st.markdown("---")
    if os.path.exists(tree_gif_path):
        try:
            st.image(tree_gif_path, caption="心之所向 ", use_container_width=True)
        except Exception:
            st.write("无法显示 Tree.gif，请确认文件是否为有效的 GIF。")
            st.markdown("<div style='color:#B8E6B0'>技能如树：从种子到枝叶，持续成长中…</div>", unsafe_allow_html=True)
    else:
        st.write("无法显示 Tree.gif，请确认文件路径：", tree_gif_path)
        st.markdown("<div style='color:#B8E6B0'>技能如树：从种子到枝叶，持续成长中…</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 中间：GIF 与简介
with middle:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("梦想去到世界的每一个角落")
    st.markdown('<div class="gif-grid">', unsafe_allow_html=True)
    for gif in gif_paths:
        try:
            st.image(gif, use_container_width=True)
        except Exception:
            st.write(f"无法显示 {gif}（请确认路径或文件名）")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 右：爱好与联系方式
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("爱好")
    for h in hobbies:
        st.markdown(f"- {h}")
    st.markdown("---")
    st.subheader("联系")
    st.markdown(f"• 推特： [{twitter}]({twitter})")
    st.markdown(f"• Instagram： [{instagram}]({instagram})")
    st.markdown(f"• 邮箱： <a href='mailto:{gmail}' style='color:#BEEBFF'>{gmail}</a>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------
# 底部：个人简介 + 炫酷科技感进度条（使用 components.html 渲染自定义进度条）
# --------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("### 个人简介")
st.markdown("本科NJFU机械电子工程（2021-2025）研究生DHU机械工程（2025-2028）梦想怎么比别人永远多拧一颗螺丝（开个玩笑）")
st.markdown("能熟练使用Matlab、Python、AutoCad、Solidworks、Abaqus")
st.markdown("---")


progress_html = """
<div style='font-family: Inter, Arial, sans-serif;'>
  <div style='display:flex; gap:18px; flex-wrap:wrap; align-items:flex-end;'>
    </div>
    <div style='flex:2; min-width:300px;'>
"""

progress_html += """
    </div>
  </div>
</div>

<style>
  .bar-fill{
    width:0%;
    background: linear-gradient(90deg, rgba(0,245,255,0.92), rgba(157,0,255,0.9), rgba(255,122,209,0.92));
    background-size: 200% 100%;
    position:relative;
    overflow:hidden;
    transition: width 1.6s cubic-bezier(.2,.9,.3,1);
  }
  .bar-fill::before{
    content:'';
    position:absolute; left:0; top:0; bottom:0; right:0;
    background-image: linear-gradient(45deg, rgba(255,255,255,0.06) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.06) 75%, transparent 75%, transparent);
    background-size: 40px 40px;
    animation: stripe-move 1.2s linear infinite;
    opacity:0.55;
  }
  .bar-fill .inner-glow{
    position:absolute; left:0; top:0; bottom:0; width:18px; background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.9), rgba(255,255,255,0.0) 40%);
    filter: blur(8px);
    mix-blend-mode: overlay;
    opacity:0.12;
  }
  @keyframes stripe-move{ from{background-position:0 0;} to{background-position:40px 0;} }
</style>

<script>
  (function(){
    const fills = document.querySelectorAll('.bar-fill');
    fills.forEach((el, i)=>{
      const pct = el.style.getPropertyValue('--pct') || window.getComputedStyle(el).getPropertyValue('--pct');
      const n = Number(String(pct).replace('%','')) || Number(pct);
      setTimeout(()=>{
        el.style.width = n + '%';
      }, 120 * i + 60);
    });

    const labels = Array.from(document.querySelectorAll('[id^="label-"]'));
    labels.forEach((lab, idx)=>{
      const target = Number(lab.textContent.replace('%','')) || 0;
      let cur = 0;
      const step = Math.max(1, Math.floor(target/24));
      const timer = setInterval(()=>{
        cur += step;
        if(cur >= target){ lab.textContent = target + '%'; clearInterval(timer); }
        else lab.textContent = cur + '%';
      }, 24 + idx*6);
    });
  })();
</script>
"""

components.html(progress_html, height=300, scrolling=False)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# 右下角柴犬（HTML + JS 注入，使其可交互）
# 如果有 dog gif 的 base64 数据就用内嵌图片，否则回退为 emoji 表示
# --------------------------
if dog_gif_data:
    shiba_inner = f'<img src="data:image/gif;base64,{dog_gif_data}" alt="shiba" class="shiba-img" />'
else:
    # 回退显示 emoji（仍然会有交互）
    shiba_inner = '<div class="shiba-img" style="display:flex; align-items:center; justify-content:center;">🐶</div>'

# 使用普通三引号字符串 + 占位符，再用 replace 注入 shiba_inner，避免 f-string 中的花括号冲突
shiba_html = """
<div id="shiba" class="shiba" title="戳一戳我～" role="button" aria-label="戳一戳柴犬">
  <div class="shiba-body" style="pointer-events:none;">{shiba_inner}</div>
  <div class="shiba-bubble">嗷！</div>
</div>

<script>
(function(){
  var shiba = document.getElementById('shiba');
  if(!shiba) return;

  var cooldown = false;

  function activateOnce(){
    if(cooldown) return;
    cooldown = true;
    shiba.classList.add('shiba-activated');

    try {
      // 在激活期间禁止再次触发（避免重复叠加）
      setTimeout(function(){
        shiba.classList.remove('shiba-activated');
        // 额外短延迟再允许触发，防止连续点击太快
        setTimeout(function(){ cooldown = false; }, 220);
      }, 1100);
    } catch(e) {
      shiba.classList.remove('shiba-activated');
      setTimeout(function(){ cooldown = false; }, 500);
    }
  }

  // 点击 / 触摸触发
  shiba.addEventListener('click', function(e){ e.stopPropagation(); activateOnce(); });
  shiba.addEventListener('touchstart', function(e){ e.stopPropagation(); activateOnce(); });

  // 鼠标按下时给出微反馈（快速放大）
  shiba.addEventListener('mousedown', function(){ shiba.style.transform = 'translateY(-3px) scale(1.03)'; });
  document.addEventListener('mouseup', function(){ shiba.style.transform = ''; });

  // 键盘无障碍（Enter / Space 激活）
  shiba.setAttribute('tabindex', '0');
  shiba.addEventListener('keydown', function(e){ if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); activateOnce(); } });

  // 防止视频背景遮挡时仍可交互（pointer-events handled in CSS/video container）
})();
</script>
"""

# 注入实际的 inner HTML（避免 f-string 中的花括号问题）
shiba_html = shiba_html.replace("{shiba_inner}", shiba_inner)

st.markdown(shiba_html, unsafe_allow_html=True)

# --------------------------
# Live2D 看板娘：左下角（改为 fixed iframe 插入，确保不随页面滚动）
# 说明：
#  - 通过在主页面插入 position:fixed 的容器并在其中放一个 srcdoc iframe 来加载 live2d-widget
#  - iframe 使用 sandbox="allow-scripts allow-same-origin" 以便脚本正常执行
#  - 这样模型会一直固定在视口左下角（与右下角柴犬同理），并保持可交互
# --------------------------

# iframe 内的完整 HTML（注意这里使用 jsonPath/have model 指向 haru 示例）
iframe_srcdoc = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  html,body{margin:0;padding:0;height:100%;background:transparent;}
  /* 确保 live2d widget 在 iframe 内显示合适 */
  .live2d-widget{position:relative !important;}
</style>
</head>
<body>
<script src="https://cdn.jsdelivr.net/gh/stevenjoezhang/live2d-widget/autoload.js"></script>
<script>
(function () {
  function initWhenReady() {
    if (typeof L2Dwidget === 'undefined') {
      setTimeout(initWhenReady, 120);
      return;
    }
    // 初始化模型（使用 haru 示例），scale 可按需调整
    L2Dwidget.init({
      model: {
        jsonPath: "https://unpkg.com/live2d-widget-model-haru@1.0.5/assets/haru.model.json",
        scale: 1
      },
      display: {
        position: "left",
        width: 240,
        height: 520,
        hOffset: 22,
        vOffset: 120
      },
      mobile: {
        show: true,
        scale: 0.6
      },
      react: {
        opacityDefault: 0.95,
        opacityOnHover: 1
      },
      log: false
    });
    // 小修正：确保 widget canvas 可交互
    setTimeout(function(){
      var w = document.querySelector('.live2d-widget');
      var c = document.querySelector('.live2d-widget-canvas');
      if(w) { w.style.pointerEvents = 'auto'; w.style.zIndex = '10010'; }
      if(c) { c.style.pointerEvents = 'auto'; }
    }, 600);
  }
  initWhenReady();
})();
</script>
</body>
</html>
"""

# 对 srcdoc 内容做 HTML 转义以安全嵌入（替换双引号）
iframe_srcdoc_escaped = html.escape(iframe_srcdoc)

# 在主页面插入一个 fixed 的容器，包含 iframe（iframe 内脚本会加载并渲染模型）
live2d_iframe_html = f'''
<div id="live2d-fixed-container" style="
  position:fixed;
  left:233px;
  bottom:80px;
  width:240px;
  height:520px;
  z-index:10010;
  pointer-events:auto;
">
  <iframe srcdoc="{iframe_srcdoc_escaped}"
          style="width:100%; height:100%; border:0; background:transparent;"
          sandbox="allow-scripts allow-same-origin">
  </iframe>
</div>
'''

# 直接把固定容器注入主 DOM（使用 st.markdown，unsafe_allow_html=True）
st.markdown(live2d_iframe_html, unsafe_allow_html=True)

# --------------- end of file ---------------
