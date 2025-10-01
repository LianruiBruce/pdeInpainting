# mask_tuner.py
# pip install opencv-python
import cv2, os
import numpy as np

img_path  = "images/cup.png"
out_path  = "masks/cup_mask.png"
init = (35, 45, 55, 65)  # 初始比例 (%)

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
H, W = img.shape[:2]
cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)

for name, val in zip(["x1","y1","x2","y2"], init):
    cv2.createTrackbar(name, "preview", val, 100, lambda _ : None)

def get_rect():
    x1 = cv2.getTrackbarPos("x1","preview")
    y1 = cv2.getTrackbarPos("y1","preview")
    x2 = cv2.getTrackbarPos("x2","preview")
    y2 = cv2.getTrackbarPos("y2","preview")
    x1, x2 = sorted([int(x1/100*W), int(x2/100*W)])
    y1, y2 = sorted([int(y1/100*H), int(y2/100*H)])
    return x1,y1,x2,y2

while True:
    x1,y1,x2,y2 = get_rect()
    overlay = img.copy()
    # 半透明可视化
    vis = overlay.copy()
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,255), -1)
    alpha = 0.25
    preview = cv2.addWeighted(vis, alpha, overlay, 1-alpha, 0)
    cv2.rectangle(preview, (x1,y1), (x2,y2), (0,255,255), 2)
    cv2.imshow("preview", preview)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('s'):
        mask = np.zeros((H,W), np.uint8)
        mask[y1:y2, x1:x2] = 255
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, mask)
        print("saved mask ->", out_path)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
