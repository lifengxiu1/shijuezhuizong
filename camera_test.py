import cv2

# 尝试不同后端，Windows 常用 DSHOW 或 MSMF
PREFS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
cap = None
for backend in PREFS:
    c = cv2.VideoCapture(0, backend)  # 如果不是 0，可试 1/2
    if c.isOpened():
        cap = c; break

# 建议 USB2.0 用 MJPG，更容易跑高分辨率
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 可改 1920x1080 或 1600x1200
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow('USB2.0_CAM1', frame)
    if cv2.waitKey(1) == 27:  # Esc 退出
        break
cap.release(); cv2.destroyAllWindows()
