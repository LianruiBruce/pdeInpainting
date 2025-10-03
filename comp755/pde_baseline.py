#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDE Inpainting Baselines (Beginner-friendly)
- OpenCV: TELEA / NAVIER
- scikit-image (optional): BIHARMONIC
- SciPy (optional): POISSON/LAPLACE inpainting (solve Δu = 0 in the hole)

Install (at least):
  pip install opencv-python
Optional:
  pip install scikit-image
  pip install scipy

Usage 1 (single file):
  python pde_inpaint_baselines.py path/to/image.png path/to/mask.png out_dir

Usage 2 (folders, paired by basename):
  python pde_inpaint_baselines.py path/to/images_dir path/to/masks_dir out_dir
"""

import os
import sys
import cv2
import numpy as np

# Optional deps
_HAS_SKI = True
try:
    from skimage.restoration import inpaint_biharmonic
except Exception:
    _HAS_SKI = False

_HAS_SCIPY = True
try:
    import scipy.sparse
    import scipy.sparse.linalg
except Exception:
    _HAS_SCIPY = False


# ------------------ basic utils ------------------

def make_dir_if_needed(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_name_without_ext(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

def read_color_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, uint8
    if img is None:
        print(f"[错误] 读不到图片: {path}")
    return img

def read_mask_image(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 0~255
    if m is None:
        print(f"[错误] 读不到 mask: {path}")
        return None
    # 强制二值化：>0 -> 255，其他 -> 0
    _, m_bin = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
    return m_bin

def save_image(path, img):
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"[错误] 保存失败: {path}")


# ------------------ Poisson (Laplace) inpainting ------------------

def inpaint_poisson(img_bgr_uint8, mask_bin_0_255):
    """
    Poisson/Laplace inpainting: solve Δu = 0 inside the hole.
    Dirichlet BC from known region.

    img_bgr_uint8: HxWx3 uint8
    mask_bin_0_255: HxW uint8, >0 is HOLE (to be filled)
    return: filled BGR uint8
    """
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy 不可用，无法运行 Poisson inpainting。")

    img = img_bgr_uint8.copy()
    h, w, c = img.shape
    hole = (mask_bin_0_255 > 0).astype(np.uint8)
    N = int(np.sum(hole))
    if N == 0:
        # 没有洞，直接返回原图
        return img

    # 像素索引映射（洞区）
    idx_map = -np.ones((h, w), dtype=np.int32)
    idx_map[hole == 1] = np.arange(N, dtype=np.int32)

    # 4-邻域方向
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 稀疏矩阵 A，右侧 b
    A = scipy.sparse.lil_matrix((N, N), dtype=np.float64)
    b = np.zeros((N, c), dtype=np.float64)

    # 构建离散拉普拉斯方程：-4*u_p + sum(u_nb) = -sum(boundary_value)
    for y in range(h):
        for x in range(w):
            if hole[y, x] == 1:
                row = idx_map[y, x]
                A[row, row] = -4.0
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if hole[ny, nx] == 1:
                            # 邻居也在洞里 -> 系数 +1
                            nb_idx = idx_map[ny, nx]
                            A[row, nb_idx] = 1.0
                        else:
                            # 邻居是边界（已知像素） -> 移到右侧
                            b[row] -= img[ny, nx, :].astype(np.float64)
                    else:
                        # 越界：当作边界值为0，通常不会出现（图像内洞）
                        pass

    A = A.tocsr()

    # 分通道解线性方程
    for ch in range(c):
        # spsolve 返回 float64
        sol = scipy.sparse.linalg.spsolve(A, b[:, ch])
        # 写回图像
        img[..., ch][hole == 1] = np.clip(sol, 0.0, 255.0).astype(np.uint8)

    return img


# ------------------ process one pair ------------------

def inpaint_one_pair(image_path, mask_path, out_dir):
    """
    Run TELEA / NAVIER / (optional) BIHARMONIC / (optional) POISSON for one image+mask.
    """
    print("--------------------------------------------------")
    print(f"[信息] 处理图片: {image_path}")
    print(f"[信息] 使用遮罩: {mask_path}")

    img = read_color_image(image_path)
    mask = read_mask_image(mask_path)
    if img is None or mask is None:
        print("[跳过] 因为读图失败")
        return

    name = get_name_without_ext(image_path)
    radius = 3  # OpenCV inpaint 半径（越大越模糊）

    # TELEA
    print("[信息] 运行 TELEA ...")
    res_telea = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    save_image(os.path.join(out_dir, f"{name}_telea.png"), res_telea)
    print(f"[完成] 保存: {os.path.join(out_dir, f'{name}_telea.png')}")

    # NAVIER–STOKES
    print("[信息] 运行 NAVIER ...")
    res_navier = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    save_image(os.path.join(out_dir, f"{name}_navier.png"), res_navier)
    print(f"[完成] 保存: {os.path.join(out_dir, f'{name}_navier.png')}")

    # BIHARMONIC (optional)
    if _HAS_SKI:
        try:
            print("[信息] 运行 BIHARMONIC ...")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_f = (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
            mask_bool = (mask.astype(np.uint8) > 0)
            try:
                out = inpaint_biharmonic(rgb_f, mask_bool, channel_axis=-1)
            except TypeError:
                out = inpaint_biharmonic(rgb_f, mask_bool, multichannel=True)
            out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            bgr_out = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
            save_image(os.path.join(out_dir, f"{name}_biharm.png"), bgr_out)
            print(f"[完成] 保存: {os.path.join(out_dir, f'{name}_biharm.png')}")
        except Exception as e:
            print(f"[警告] Biharmonic 失败，跳过：{e}")
    else:
        print("[提示] 未安装 scikit-image，跳过 Biharmonic（可选安装：pip install scikit-image）")

    # POISSON (optional)
    if _HAS_SCIPY:
        try:
            print("[信息] 运行 POISSON ...")
            res_poisson = inpaint_poisson(img.copy(), mask)
            save_image(os.path.join(out_dir, f"{name}_poisson.png"), res_poisson)
            print(f"[完成] 保存: {os.path.join(out_dir, f'{name}_poisson.png')}")
        except Exception as e:
            print(f"[警告] Poisson 失败，跳过：{e}")
    else:
        print("[提示] 未安装 SciPy，跳过 Poisson（可选安装：pip install scipy）")


# ------------------ folder pairing ------------------

def build_mask_map(mask_dir):
    m = {}
    for fn in os.listdir(mask_dir):
        full = os.path.join(mask_dir, fn)
        if os.path.isfile(full):
            m[get_name_without_ext(full)] = full
    return m

def process_folder_pair(image_dir, mask_dir, out_dir):
    make_dir_if_needed(out_dir)
    mask_map = build_mask_map(mask_dir)

    total = 0
    matched = 0

    for fn in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fn)
        if not os.path.isfile(img_path):
            continue
        total += 1
        key = get_name_without_ext(img_path)
        if key in mask_map:
            matched += 1
            inpaint_one_pair(img_path, mask_map[key], out_dir)
        else:
            print(f"[警告] 找不到同名 mask: {key}.*")

    print("==================================================")
    print(f"[汇总] 共发现图片文件: {total}")
    print(f"[汇总] 成功配对数量:   {matched}")
    print(f"[输出] 输出文件夹:     {out_dir}")


# ------------------ main ------------------

def main():
    if len(sys.argv) < 4:
        print("用法：")
        print("  单文件：python pde_inpaint_baselines.py <image.png> <mask.png> <out_dir>")
        print("  文件夹：python pde_inpaint_baselines.py <images_dir> <masks_dir> <out_dir>")
        print("说明：mask 中 >0 的像素会被当成“洞”进行修补。")
        sys.exit(0)

    img_input = sys.argv[1]
    mask_input = sys.argv[2]
    out_dir = sys.argv[3]
    make_dir_if_needed(out_dir)

    if os.path.isfile(img_input):
        if not os.path.isfile(mask_input):
            print("[错误] 你给的是单张图片，但 mask 不是文件路径。")
            sys.exit(1)
        inpaint_one_pair(img_input, mask_input, out_dir)
    else:
        if not os.path.isdir(img_input) or not os.path.isdir(mask_input):
            print("[错误] 你想用文件夹模式，但有路径不是文件夹。")
            sys.exit(1)
        process_folder_pair(img_input, mask_input, out_dir)

if __name__ == "__main__":
    main()
