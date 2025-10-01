# gen_masks.py
# 作用：为 images/ 中的每张图片生成一个同尺寸、纯黑底白洞的 PNG 掩码，
# 保存到 masks/ 目录，文件名固定为  <原图名>_mask.png （若存在则覆盖）。

import os, glob, argparse, random
import numpy as np
from PIL import Image, ImageDraw

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def rect_mask(w, h, coverage=0.12, rng=None):
    """单个随机矩形洞；coverage 为目标占比（0~1）"""
    rng = rng or random
    area = w * h
    target = max(16, int(area * coverage))
    rw = max(8, int(np.sqrt(target) * rng.uniform(0.7, 1.3)))
    rh = max(8, int(target / max(rw, 8)))
    rw = min(rw, w-2); rh = min(rh, h-2)
    x = rng.randint(1, max(1, w - rw - 1))
    y = rng.randint(1, max(1, h - rh - 1))
    m = Image.new("L", (w, h), 0)             # 黑底
    ImageDraw.Draw(m).rectangle([x, y, x+rw, y+rh], fill=255)  # 白洞
    return m

def stroke_mask(w, h, strokes=2, rng=None):
    """笔刷涂抹式洞"""
    rng = rng or random
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for _ in range(strokes):
        pts = [(rng.randint(0, w-1), rng.randint(0, h-1)) for _ in range(rng.randint(3,6))]
        width = max(6, int(min(w, h) * rng.uniform(0.01, 0.03)))
        d.line(pts, fill=255, width=width, joint="curve")
    return m

def mixed_mask(w, h, rng=None, coverage=0.1):
    """矩形 + 笔刷合成"""
    rng = rng or random
    a = np.array(rect_mask(w, h, coverage, rng))
    b = np.array(stroke_mask(w, h, strokes=1, rng=rng))
    return Image.fromarray(np.maximum(a, b))   # 合并为白洞

def make_mask(img_path, out_path, mode="rect", seed=None, coverage=0.12):
    rng = random.Random(seed)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if mode == "rect":
        m = rect_mask(w, h, coverage, rng)
    elif mode == "stroke":
        m = stroke_mask(w, h, rng=rng)
    else:  # mixed
        m = mixed_mask(w, h, rng=rng, coverage=coverage)
    m.save(out_path, "PNG")  # 固定覆盖写出

def main():
    ap = argparse.ArgumentParser(description="Generate one binary mask per image (white=hole, black=keep).")
    ap.add_argument("--images", default="images", help="folder of input images")
    ap.add_argument("--out",    default="masks",  help="folder to save masks")
    ap.add_argument("--mode",   default="rect",   choices=["rect","stroke","mixed"])
    ap.add_argument("--coverage", type=float, default=0.12, help="area ratio for rectangle mask")
    ap.add_argument("--seed", type=int, default=None, help="global seed for reproducibility")
    args = ap.parse_args()

    ensure_dir(args.out)
    paths = sorted(glob.glob(os.path.join(args.images, "*.*")))
    if not paths:
        print(f"No images found in {args.images}"); return

    base_rng = random.Random(args.seed)
    for p in paths:
        name, _ = os.path.splitext(os.path.basename(p))
        out = os.path.join(args.out, f"{name}_mask.png")   # 固定命名，覆盖
        # 给每张图一个确定的子种子，保证可复现
        seed_i = None if args.seed is None else base_rng.randint(0, 2**31-1)
        make_mask(p, out, mode=args.mode, seed=seed_i, coverage=args.coverage)
        print("saved", out)

if __name__ == "__main__":
    main()
