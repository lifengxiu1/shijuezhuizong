import cv2
import numpy as np
import time
import mediapipe as mp

# ====== 新增：SO100 机器人相关（适配 lerobot 0.4.1 的路径） ======
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig


# ---------- 绘制辅助 ----------
def draw_gesture_status(img, is_tap, is_hold):
    h, w, _ = img.shape
    if is_tap:
        text, color = "TAP!", (0, 255, 255)      # 黄
    elif is_hold:
        text, color = "HOLD!", (255, 0, 255)     # 洋红
    else:
        text, color = "Watching...", (255, 255, 255)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    tips = [
        "Pinch index & thumb:",
        "Quick pinch = TAP (yellow)",
        "Long pinch = HOLD (magenta)",
        "Press 'q' to quit",
    ]
    for i, t in enumerate(tips):
        cv2.putText(img, t, (10, h - 100 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_hand(img, palm_center, is_tap=False, is_hold=False):
    h, w, _ = img.shape
    # 传进来是[-1,1]坐标，这里转像素并考虑镜像
    px = int(w - ((-palm_center[0] + 1) / 2) * w)
    py = int(((palm_center[1] + 1) / 2) * h)
    color = (0, 0, 255)
    if is_tap:
        color = (0, 255, 255)
    elif is_hold:
        color = (255, 0, 255)
    cv2.circle(img, (px, py), 8, color, -1)


def draw_finger_tips(img, tracker, is_hold=False):
    if tracker.index_pos is None or tracker.thumb_pos is None:
        return
    h, w, _ = img.shape
    ix = int(w - tracker.index_pos[0] * w)
    iy = int(tracker.index_pos[1] * h)
    tx = int(w - tracker.thumb_pos[0] * w)
    ty = int(tracker.thumb_pos[1] * h)
    index_px, thumb_px = (ix, iy), (tx, ty)

    cv2.circle(img, index_px, 8, (0, 255, 0), -1)     # 食指
    cv2.circle(img, thumb_px, 8, (255, 255, 0), -1)   # 拇指
    cv2.line(img, index_px, thumb_px, (255, 255, 255), 2)

    mid_x, mid_y = (ix + tx) // 2, (iy + ty) // 2
    if tracker.current_distance is not None:
        cv2.putText(img, f"Distance: {tracker.current_distance:.3f}",
                    (mid_x - 50, mid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(img, f"Duration: {tracker.current_duration:.2f}s",
                    (mid_x - 50, mid_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    cv2.putText(img, "INDEX", (ix - 30, iy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, "THUMB", (tx - 30, ty - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if is_hold:
        center = (w // 2, h // 2)
        cv2.line(img, center, (mid_x, mid_y), (255, 0, 0), 3)
        cv2.circle(img, center, 5, (255, 0, 0), -1)
        cv2.circle(img, (mid_x, mid_y), 5, (255, 0, 0), -1)
        err = np.hypot(mid_x - center[0], mid_y - center[1])
        cv2.putText(img, f"Error: {err:.1f}px", (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# ---------- 手势逻辑 ----------
mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self, nb_hands=1, tap_threshold=0.25, hold_threshold=0.6, distance_threshold=0.05):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=nb_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.tap_threshold = tap_threshold
        self.hold_threshold = hold_threshold
        self.distance_threshold = distance_threshold
        self.finger_down_start = None
        self.is_finger_down = False

        self.current_distance = None
        self.current_duration = 0.0
        self.index_pos = None
        self.thumb_pos = None
        self.just_tapped = False

    def get_palm_centers(self, img):
        img = cv2.flip(img, 1)
        res = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks or not res.multi_handedness:
            return None
        centers = []
        for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            if handed.classification[0].label != 'Right':
                continue
            pip = lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            centers.append([-(pip.x - 0.5) * 2, (pip.y - 0.5) * 2])
        return centers or None

    def update(self, img):
        img = cv2.flip(img, 1)
        res = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        now = time.time()
        self.just_tapped = False

        if res.multi_hand_landmarks and res.multi_handedness:
            for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                if handed.classification[0].label != 'Right':
                    continue
                idx = lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thb = lms.landmark[mp_hands.HandLandmark.THUMB_TIP]
                self.index_pos = np.array([idx.x, idx.y])
                self.thumb_pos = np.array([thb.x, thb.y])
                self.current_distance = float(np.linalg.norm(self.index_pos - self.thumb_pos))

                if not self.is_finger_down and self.current_distance < self.distance_threshold:
                    self.is_finger_down = True
                    self.finger_down_start = now

                if self.is_finger_down:
                    self.current_duration = (now - self.finger_down_start) if self.finger_down_start else 0.0
                    if self.current_distance > self.distance_threshold:
                        if self.current_duration < self.tap_threshold:
                            self.just_tapped = True
                        self._reset()
                else:
                    self.current_duration = 0.0
                return  # 只取第一只右手

        self._reset()
        self.current_distance = None
        self.index_pos = None
        self.thumb_pos = None

    def isTap(self):
        return self.just_tapped

    def isHold(self):
        return (
            self.is_finger_down and
            self.current_distance is not None and
            self.current_distance < self.distance_threshold and
            self.current_duration >= self.hold_threshold
        )

    def _reset(self):
        self.finger_down_start = None
        self.is_finger_down = False
        self.current_duration = 0.0


# ====== SO100 相关配置 ======
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def init_robot():
    """
    初始化 SO100 从臂：
    - 使用端口 COM3（你前面校准用的那个）
    - id 留空（None），对应刚刚生成的 None.json 校准文件
    """
    cfg = SO100FollowerConfig(
        port="COM3",   # 如串口变了，这里改
        # 其他字段使用默认值：id=None, cameras={}, use_degrees=False 等
    )
    robot = SO100Follower(cfg)
    # 校准已经通过 lerobot-calibrate 做过了，这里只需要 connect
    robot.connect()
    print("✅ SO100 Follower 已连接 (COM3)")
    return robot


# ---------- 主循环 ----------
def main():
    # 先连上机械臂
    robot = init_robot()

    CAM_INDEX = 1      # 如果你的 USB 摄像头是 1/2，就改成 1/2
    WIDTH, HEIGHT = 1280, 720

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("无法打开摄像头")
        robot.disconnect()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = HandTracker(
        nb_hands=1,
        tap_threshold=0.5,
        hold_threshold=0.6,
        distance_threshold=0.05
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tracker.update(frame)
        is_tap = tracker.isTap()
        is_hold = tracker.isHold()

        # ---------- 读取当前关节状态 ----------
        try:
            obs = robot.get_observation()
            current = np.array([obs[name] for name in JOINT_NAMES], dtype=np.float32)
        except Exception as e:
            print("读取机械臂状态失败：", e)
            current = None

        # ---------- 用“捏住”手势拖动机械臂 ----------
        if (
            current is not None
            and is_hold
            and tracker.index_pos is not None
            and tracker.thumb_pos is not None
        ):
            h, w, _ = frame.shape

            # 手指中点（0~1 坐标）
            mid_x = (tracker.index_pos[0] + tracker.thumb_pos[0]) / 2.0
            mid_y = (tracker.index_pos[1] + tracker.thumb_pos[1]) / 2.0

            # 映射到像素坐标（按你之前的坐标系做左右镜像）
            px = int((1.0 - mid_x) * w)
            py = int(mid_y * h)

            # 相对图像中心的偏移，范围大致 [-1, 1]
            cx, cy = w // 2, h // 2
            off_x = (px - cx) / cx      # 右正
            off_y = (py - cy) / cy      # 下正

            # 将偏移映射成关节的小增量（单位：电机刻度）
            gain = 80.0  # 越大动得越快；如太猛可改成 30 / 50
            delta = np.zeros_like(current)
            delta[0] = gain * off_x       # shoulder_pan
            delta[1] = -gain * off_y      # shoulder_lift（手往上 → 机械臂抬起）

            target = current + delta

            action = {
                name: float(target[i])
                for i, name in enumerate(JOINT_NAMES)
            }

            try:
                robot.send_action(action)
            except Exception as e:
                print("发送动作失败：", e)

        centers = tracker.get_palm_centers(frame)
        if centers:
            draw_hand(frame, centers[0], is_tap, is_hold)

        draw_gesture_status(frame, is_tap, is_hold)
        draw_finger_tips(frame, tracker, is_hold)

        cv2.imshow("Gesture (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    robot.disconnect()
    print("🔌 已断开 SO100 Follower")


if __name__ == "__main__":
    main()
