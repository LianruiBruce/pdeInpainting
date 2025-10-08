#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2, os
import numpy as np

img_path  = "images/cup.png"
out_path  = "masks/cup.png"
init = (35, 45, 55, 65)            # 初始比例 (%), 必须在 [0,100]
min_size_pct = 1                   # 最小宽/高占比，避免零面积
preview_max_w = 1200               # 预览最大宽度（仅影响显示，不影响保存）

# --- 读图与检查 ---
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")
H, W = img.shape[:2]

win_name = "preview (s=save, q=quit)"
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

# --- 轨迹条 ---
def _noop(_): pass
for name, val in zip(["x1","y1","x2","y2"], init):
    v = int(np.clip(val, 0, 100))
    cv2.createTrackbar(name, win_name, v, 100, _noop)

def get_rect_pixels():
    x1 = cv2.getTrackbarPos("x1", win_name)
    y1 = cv2.getTrackbarPos("y1", win_name)
    x2 = cv2.getTrackbarPos("x2", win_name)
    y2 = cv2.getTrackbarPos("y2", win_name)
    x1, x2 = sorted([int(x1/100.0*W), int(x2/100.0*W)])
    y1, y2 = sorted([int(y1/100.0*H), int(y2/100.0*H)])
    # 最小尺寸约束
    min_w = max(1, int(min_size_pct/100.0*W))
    min_h = max(1, int(min_size_pct/100.0*H))
    if x2 - x1 < min_w: x2 = min(W-1, x1 + min_w)
    if y2 - y1 < min_h: y2 = min(H-1, y1 + min_h)
    return x1, y1, x2, y2

# 计算仅用于显示的缩放比例
scale = 1.0
if W > preview_max_w:
    scale = preview_max_w / float(W)

print("Controls: 's' to save mask, 'q' to quit.")
print(f"Image size: {W}x{H}, preview scale: {scale:.2f}x")

while True:
    # 如果窗口被手动关闭，跳出
    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    x1, y1, x2, y2 = get_rect_pixels()
    overlay = img.copy()

    # 半透明可视化
    vis = overlay.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), -1)
    alpha = 0.25
    preview = cv2.addWeighted(vis, alpha, overlay, 1 - alpha, 0)
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # 仅显示时缩放
    if scale != 1.0:
        preview_disp = cv2.resize(preview, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    else:
        preview_disp = preview

    cv2.imshow(win_name, preview_disp)
    k = cv2.waitKey(30) & 0xFF

    if k == ord('s') or k == ord('S'):
        mask = np.zeros((H, W), np.uint8)
        mask[y1:y2, x1:x2] = 255
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if cv2.imwrite(out_path, mask):
            print(f"Saved mask -> {out_path}  (rect=({x1},{y1})-({x2},{y2}))")
        else:
            print(f"[ERROR] Failed to save mask -> {out_path}")

    if k == ord('q') or k == 27:  # q 或 ESC
        break

cv2.destroyAllWindows()
