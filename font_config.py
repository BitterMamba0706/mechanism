# font_config.py
import matplotlib.pyplot as plt
from matplotlib import font_manager

def set_chinese_font():
    """自动寻找并设置 matplotlib 中文字体"""
    zh_fonts = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"
    ]
    font_path = None
    for f in zh_fonts:
        try:
            font_path = font_manager.findfont(f, fallback_to_default=False)
            if font_path:
                break
        except Exception:
            continue

    if font_path:
        plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=font_path).get_name()]
        plt.rcParams['axes.unicode_minus'] = False
    else:
        # 找不到中文字体时的兜底
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️ 未找到中文字体，可能会乱码")
