#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDE Inpainting Baselines (Beginner-friendly)
- OpenCV: TELEA / NAVIER-STOKES
- scikit-image: BIHARMONIC
- SciPy: POISSON/LAPLACE inpainting (solve Δu = 0 in the hole)

Usage 1 (single file):
  python pde_inpaint_baselines.py path/to/image.png path/to/mask.png out_dir

Usage 2 (folders, paired by basename):
  python pde_inpaint_baselines.py path/to/images_dir path/to/masks_dir out_dir

Note: Mask pixels > 0 are treated as holes to be filled.
"""

import os
import sys
import cv2
import numpy as np

# Optional dependencies
_HAS_SKI = True
try:
    from skimage.restoration import inpaint_biharmonic
except ImportError:
    _HAS_SKI = False

_HAS_SCIPY = True
try:
    import scipy.sparse
    import scipy.sparse.linalg
except ImportError:
    _HAS_SCIPY = False


# ------------------ Basic Utilities ------------------

def make_dir_if_needed(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_name_without_ext(path):
    """Extract filename without extension from path."""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

def read_color_image(path):
    """Read color image in BGR format."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, uint8
    if img is None:
        print(f"[ERROR] Cannot read image: {path}")
    return img

def read_mask_image(path):
    """Read and binarize mask image."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 0~255
    if m is None:
        print(f"[ERROR] Cannot read mask: {path}")
        return None
    # Force binarization: >0 -> 255, else -> 0
    _, m_bin = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
    return m_bin

def save_image(path, img):
    """Save image to disk."""
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"[ERROR] Failed to save: {path}")
    return ok


# ------------------ Poisson (Laplace) Inpainting ------------------

def inpaint_poisson(img_bgr_uint8, mask_bin_0_255):
    """
    Poisson/Laplace inpainting: solve Δu = 0 inside the hole.
    Uses Dirichlet boundary conditions from known regions.

    Args:
        img_bgr_uint8: HxWx3 uint8 BGR image
        mask_bin_0_255: HxW uint8 mask, >0 indicates HOLE (to be filled)
    
    Returns:
        Inpainted BGR uint8 image
    """
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy is not available. Cannot run Poisson inpainting.")

    img = img_bgr_uint8.copy()
    h, w, c = img.shape
    hole = (mask_bin_0_255 > 0).astype(np.uint8)
    N = int(np.sum(hole))
    
    if N == 0:
        # No holes to fill, return original image
        return img

    # Pixel index mapping for hole region
    idx_map = -np.ones((h, w), dtype=np.int32)
    idx_map[hole == 1] = np.arange(N, dtype=np.int32)

    # 4-neighborhood directions
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Sparse matrix A and right-hand side b
    A = scipy.sparse.lil_matrix((N, N), dtype=np.float64)
    b = np.zeros((N, c), dtype=np.float64)

    # Build discrete Laplace equation: -4*u_p + sum(u_nb) = -sum(boundary_values)
    for y in range(h):
        for x in range(w):
            if hole[y, x] == 1:
                row = idx_map[y, x]
                A[row, row] = -4.0
                
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if hole[ny, nx] == 1:
                            # Neighbor is also in hole -> coefficient +1
                            nb_idx = idx_map[ny, nx]
                            A[row, nb_idx] = 1.0
                        else:
                            # Neighbor is boundary (known pixel) -> move to RHS
                            b[row] -= img[ny, nx, :].astype(np.float64)

    A = A.tocsr()

    # Solve linear system for each channel
    for ch in range(c):
        sol = scipy.sparse.linalg.spsolve(A, b[:, ch])
        # Write solution back to image
        img[..., ch][hole == 1] = np.clip(sol, 0.0, 255.0).astype(np.uint8)

    return img


# ------------------ Process One Image-Mask Pair ------------------

def inpaint_one_pair(image_path, mask_path, out_dir):
    """
    Apply multiple inpainting methods to one image-mask pair.
    Methods: TELEA, NAVIER-STOKES, BIHARMONIC, POISSON
    """
    print("=" * 50)
    print(f"[INFO] Processing image: {image_path}")
    print(f"[INFO] Using mask: {mask_path}")

    img = read_color_image(image_path)
    mask = read_mask_image(mask_path)
    
    if img is None or mask is None:
        print("[SKIP] Failed to read image or mask")
        return

    name = get_name_without_ext(image_path)
    radius = 3  # OpenCV inpaint radius (larger = more blur)

    # Method 1: TELEA
    print("[INFO] Running TELEA inpainting...")
    res_telea = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    out_path = os.path.join(out_dir, f"{name}_telea.png")
    if save_image(out_path, res_telea):
        print(f"[DONE] Saved: {out_path}")

    # Method 2: NAVIER-STOKES
    print("[INFO] Running NAVIER-STOKES inpainting...")
    res_navier = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    out_path = os.path.join(out_dir, f"{name}_navier.png")
    if save_image(out_path, res_navier):
        print(f"[DONE] Saved: {out_path}")

    # Method 3: BIHARMONIC (optional)
    if _HAS_SKI:
        try:
            print("[INFO] Running BIHARMONIC inpainting...")
            # Convert BGR to RGB and normalize to [0, 1]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_f = (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
            mask_bool = (mask.astype(np.uint8) > 0)
            
            # Handle different scikit-image versions
            try:
                out = inpaint_biharmonic(rgb_f, mask_bool, channel_axis=-1)
            except TypeError:
                out = inpaint_biharmonic(rgb_f, mask_bool, multichannel=True)
            
            # Convert back to BGR uint8
            out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            bgr_out = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(out_dir, f"{name}_biharmonic.png")
            if save_image(out_path, bgr_out):
                print(f"[DONE] Saved: {out_path}")
        except Exception as e:
            print(f"[WARNING] Biharmonic inpainting failed: {e}")
    else:
        print("[INFO] scikit-image not installed, skipping Biharmonic")
        print("       (Optional: pip install scikit-image)")

    # Method 4: POISSON (optional)
    if _HAS_SCIPY:
        try:
            print("[INFO] Running POISSON inpainting...")
            res_poisson = inpaint_poisson(img.copy(), mask)
            out_path = os.path.join(out_dir, f"{name}_poisson.png")
            if save_image(out_path, res_poisson):
                print(f"[DONE] Saved: {out_path}")
        except Exception as e:
            print(f"[WARNING] Poisson inpainting failed: {e}")
    else:
        print("[INFO] SciPy not installed, skipping Poisson")
        print("       (Optional: pip install scipy)")


# ------------------ Folder Processing ------------------

def build_mask_map(mask_dir):
    """Build a dictionary mapping basename to mask file path."""
    mask_map = {}
    for filename in os.listdir(mask_dir):
        full_path = os.path.join(mask_dir, filename)
        if os.path.isfile(full_path):
            mask_map[get_name_without_ext(full_path)] = full_path
    return mask_map

def process_folder_pair(image_dir, mask_dir, out_dir):
    """Process all matching image-mask pairs in folders."""
    make_dir_if_needed(out_dir)
    mask_map = build_mask_map(mask_dir)

    total_images = 0
    matched_pairs = 0

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        if not os.path.isfile(img_path):
            continue
        
        total_images += 1
        basename = get_name_without_ext(img_path)
        
        if basename in mask_map:
            matched_pairs += 1
            inpaint_one_pair(img_path, mask_map[basename], out_dir)
        else:
            print(f"[WARNING] No matching mask found for: {basename}.*")

    print("=" * 50)
    print(f"[SUMMARY] Total image files found: {total_images}")
    print(f"[SUMMARY] Successfully matched pairs: {matched_pairs}")
    print(f"[SUMMARY] Output directory: {out_dir}")


# ------------------ Main Entry Point ------------------

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 4:
        print("Usage:")
        print("  Single file: python pde_inpaint_baselines.py <image.png> <mask.png> <out_dir>")
        print("  Folder mode: python pde_inpaint_baselines.py <images_dir> <masks_dir> <out_dir>")
        print("\nNote: Mask pixels > 0 are treated as holes to be inpainted.")
        sys.exit(0)

    img_input = sys.argv[1]
    mask_input = sys.argv[2]
    out_dir = sys.argv[3]
    
    make_dir_if_needed(out_dir)

    # Check if single file or folder mode
    if os.path.isfile(img_input):
        if not os.path.isfile(mask_input):
            print("[ERROR] Image is a file but mask is not a valid file path.")
            sys.exit(1)
        inpaint_one_pair(img_input, mask_input, out_dir)
    else:
        if not os.path.isdir(img_input) or not os.path.isdir(mask_input):
            print("[ERROR] For folder mode, both paths must be directories.")
            sys.exit(1)
        process_folder_pair(img_input, mask_input, out_dir)

    print("\n[INFO] Processing complete!")


if __name__ == "__main__":
    main()