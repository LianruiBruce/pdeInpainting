import os, sys, glob
import numpy as np

# --- 依赖尽量少：OpenCV 必要；skimage 可选（没有就只跑前两种） ---
import cv2
try:
    from skimage.restoration import inpaint_biharmonic
    _HAS_SKI = True
except Exception:
    _HAS_SKI = False

def run_one(img_path, mask_path, out_dir, methods=("telea","navier","biharm")):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(img_path))[0]

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)       # BGR uint8
    if img is None:
        print(f"[skip] cannot read {img_path}"); return
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 0/255
    if mask is None:
        print(f"[skip] cannot read {mask_path}"); return

    # OpenCV 约定：mask>0 为洞
    if "telea" in methods:
        res = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(out_dir, f"{name}_telea.png"), res)
    if "navier" in methods:
        res = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        cv2.imwrite(os.path.join(out_dir, f"{name}_navier.png"), res)
    if "biharm" in methods and _HAS_SKI:
        # skimage 需要 RGB & float 且 mask 为 bool
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mb   = mask.astype(np.uint8) > 0
        out  = inpaint_biharmonic(rgb.astype(np.float32)/255.0, mb, channel_axis=2)
        bgr  = cv2.cvtColor((out*255.0).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{name}_biharm.png"), bgr)

def main():
    if len(sys.argv) < 4:
        print("usage:")
        print("  python pde_baseline.py <img_path_or_dir> <mask_path_or_dir> <out_dir>")
        print("notes: mask>0 treated as hole (to be filled). Filenames should match when using dirs.")
        sys.exit(0)

    img_in, mask_in, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    # 情况1：单文件
    if os.path.isfile(img_in):
        mask_path = mask_in if os.path.isfile(mask_in) else None
        if not mask_path:
            print("mask file not found"); sys.exit(1)
        run_one(img_in, mask_path, out_dir)
        return

    # 情况2：文件夹（按同名文件配对）
    imgs  = sorted(glob.glob(os.path.join(img_in, "*.*")))
    masks = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(mask_in, "*.*"))}
    for ip in imgs:
        key = os.path.splitext(os.path.basename(ip))[0]
        if key in masks:
            run_one(ip, masks[key], out_dir)

if __name__ == "__main__":
    main()
