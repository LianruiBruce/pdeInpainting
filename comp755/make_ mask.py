# make_mask.py
# pip install pillow
from PIL import Image, ImageDraw
import random

def gen_mask(img_path, out_path,
             mode="rect",
             rect_norm=(0.35, 0.45, 0.55, 0.65),  # 相对坐标 (x1,y1,x2,y2), 0~1
             strokes=2, width_ratio=(0.015, 0.03), seed=42):
    """
    生成与 img 同尺寸的二值掩码(黑底白洞)：
      - mode="rect"  用矩形洞，位置由 rect_norm 控制（相对比例）
      - mode="stroke"  用笔刷涂抹洞，数量 strokes，线宽按 width_ratio 比例
    """
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    mask = Image.new("L", (W, H), 0)      # 黑底
    draw = ImageDraw.Draw(mask)

    if mode == "rect":
        x1 = int(rect_norm[0] * W); y1 = int(rect_norm[1] * H)
        x2 = int(rect_norm[2] * W); y2 = int(rect_norm[3] * H)
        draw.rectangle([x1, y1, x2, y2], fill=255)  # 白洞
    else:  # stroke
        rnd = random.Random(seed)
        wmin = int(min(W, H) * width_ratio[0])
        wmax = int(min(W, H) * width_ratio[1])
        for _ in range(strokes):
            npts = rnd.randint(3, 6)
            pts = [(rnd.randint(0, W-1), rnd.randint(0, H-1)) for _ in range(npts)]
            width = rnd.randint(max(3, wmin), max(wmin+1, wmax))
            draw.line(pts, fill=255, width=width, joint="curve")

    mask.save(out_path, "PNG")
    print(f"saved mask -> {out_path} (white=hole, black=keep)")

if __name__ == "__main__":
    # === 按你的杯子图大致位置生成矩形洞 ===
    gen_mask(
        img_path="images/cup.png",           # 原图路径
        out_path="masks/cup_mask.png",       # 输出mask路径(PNG)
        mode="rect",                         # 或 "stroke"
        rect_norm=(0.35, 0.45, 0.55, 0.65),  # 可微调位置/大小
        # strokes=2,                         # 若 mode="stroke" 可调笔刷条数
        # width_ratio=(0.015, 0.03),         # 若 stroke 可调线宽比例
        seed=42
    )
