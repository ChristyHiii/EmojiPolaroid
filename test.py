# file: face_emoji_cam.py
"""
实时摄像头人脸表情识别（基于 MediaPipe FaceMesh 的规则判定）并在脸上叠加绘制版 emoji。
依赖: mediapipe, opencv-python, numpy
安装: pip install mediapipe opencv-python numpy
运行: python face_emoji_cam.py
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Tuple, List

# ---- 少量工具函数 ----
def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def to_px(point, w, h):
    return int(point[0] * w), int(point[1] * h)

# ---- 绘制一个简易的“emoji”图案（使用 OpenCV 绘制） ----
def draw_emoji(canvas, center: Tuple[int,int], size: int, expression: str):
    """
    canvas: BGR image
    center: (x, y) center of emoji
    size: diameter in pixels
    expression: 'happy','surprised','angry','sad','neutral'
    """
    x, y = center
    r = int(size // 2)
    # Face circle color
    face_color = (0, 215, 255)  # BGR (yellowish)
    thickness = -1
    cv2.circle(canvas, (x, y), r, face_color, thickness)

    eye_radius = max(1, r // 6)
    eye_y_offset = -r // 5
    eye_x_offset = r // 2 - eye_radius

    left_eye_c = (x - eye_x_offset, y + eye_y_offset)
    right_eye_c = (x + eye_x_offset, y + eye_y_offset)

    # Draw eyes depending on expression
    if expression == 'happy':
        # simple filled eyes and smiling mouth
        cv2.circle(canvas, left_eye_c, eye_radius, (0, 0, 0), -1)
        cv2.circle(canvas, right_eye_c, eye_radius, (0, 0, 0), -1)
        # smile arc
        start = (x - r//2, y + r//6)
        end = (x + r//2, y + r//6)
        axes = (r//2, r//3)
        cv2.ellipse(canvas, (x, y + r//6), axes, 0, 20, 160, (0,0,0), r//12)
    elif expression == 'surprised':
        # big open round mouth and wide eyes
        cv2.circle(canvas, left_eye_c, eye_radius+1, (255,255,255), -1)
        cv2.circle(canvas, right_eye_c, eye_radius+1, (255,255,255), -1)
        cv2.circle(canvas, left_eye_c, eye_radius//2, (0,0,0), -1)
        cv2.circle(canvas, right_eye_c, eye_radius//2, (0,0,0), -1)
        # mouth circle
        mouth_r = max(3, r//4)
        cv2.circle(canvas, (x, y + r//6), mouth_r, (0,0,0), -1)
    elif expression == 'angry':
        # slanted eyebrows, small eyes, straight/angry mouth
        # eyebrows (drawn as lines)
        brow_offset_y = y - r//3
        cv2.line(canvas, (left_eye_c[0]-eye_radius, brow_offset_y+8), (left_eye_c[0]+eye_radius+6, brow_offset_y-6), (0,0,0), r//15)
        cv2.line(canvas, (right_eye_c[0]+eye_radius, brow_offset_y-6), (right_eye_c[0]-eye_radius-6, brow_offset_y+8), (0,0,0), r//15)
        cv2.circle(canvas, left_eye_c, max(1, eye_radius//2), (0,0,0), -1)
        cv2.circle(canvas, right_eye_c, max(1, eye_radius//2), (0,0,0), -1)
        # mouth: straight line or slight frown
        cv2.line(canvas, (x - r//3, y + r//6), (x + r//3, y + r//6 + r//12), (0,0,0), r//12)
    elif expression == 'sad':
        # small eyes and frown
        cv2.circle(canvas, left_eye_c, max(1, eye_radius//2), (0,0,0), -1)
        cv2.circle(canvas, right_eye_c, max(1, eye_radius//2), (0,0,0), -1)
        # frown arc (upside down)
        cv2.ellipse(canvas, (x, y + r//3), (r//2, r//4), 0, 200, 340, (0,0,0), r//12)
    else:  # neutral
        cv2.circle(canvas, left_eye_c, max(1, eye_radius//2), (0,0,0), -1)
        cv2.circle(canvas, right_eye_c, max(1, eye_radius//2), (0,0,0), -1)
        # straight mouth
        cv2.line(canvas, (x - r//4, y + r//6), (x + r//4, y + r//6), (0,0,0), r//15)

# ---- 表情判定规则 ----
def estimate_expression(landmarks: List[Tuple[float,float]], img_w: int, img_h: int) -> str:
    """
    landmarks: list of (x,y) normalized coords from FaceMesh
    返回: 'happy','surprised','angry','sad','neutral'
    """
    # Keypoint indices (MediaPipe FaceMesh common indices)
    # mouth: left 61, right 291, upper 13, lower 14
    # left eye: top 159, bottom 145, left 33, right 133
    # right eye: top 386, bottom 374, left 362, right 263
    # left eyebrow approximate: 105; right eyebrow: 334
    idx = {}
    get = lambda i: (landmarks[i][0], landmarks[i][1])
    try:
        mouth_l = get(61)
        mouth_r = get(291)
        mouth_top = get(13)
        mouth_bottom = get(14)
        left_eye_top = get(159)
        left_eye_bot = get(145)
        left_eye_left = get(33)
        left_eye_right = get(133)
        right_eye_top = get(386)
        right_eye_bot = get(374)
        right_eye_left = get(362)
        right_eye_right = get(263)
        left_brow = get(105)
        right_brow = get(334)
    except Exception:
        return 'neutral'

    # convert to pixel coords for scale normalization
    mouth_l_px = (mouth_l[0]*img_w, mouth_l[1]*img_h)
    mouth_r_px = (mouth_r[0]*img_w, mouth_r[1]*img_h)
    mouth_top_px = (mouth_top[0]*img_w, mouth_top[1]*img_h)
    mouth_bot_px = (mouth_bottom[0]*img_w, mouth_bottom[1]*img_h)

    # face bbox size for normalization
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    face_w = max(1e-6, (max(xs)-min(xs))*img_w)
    face_h = max(1e-6, (max(ys)-min(ys))*img_h)

    mouth_width = dist(mouth_l_px, mouth_r_px)
    mouth_height = dist(mouth_top_px, mouth_bot_px)
    mouth_open_ratio = mouth_height / face_h
    smile_ratio = mouth_width / (mouth_height + 1e-6)

    # eyes
    l_eye_top_px = (left_eye_top[0]*img_w, left_eye_top[1]*img_h)
    l_eye_bot_px = (left_eye_bot[0]*img_w, left_eye_bot[1]*img_h)
    l_eye_left_px = (left_eye_left[0]*img_w, left_eye_left[1]*img_h)
    l_eye_right_px = (left_eye_right[0]*img_w, left_eye_right[1]*img_h)

    r_eye_top_px = (right_eye_top[0]*img_w, right_eye_top[1]*img_h)
    r_eye_bot_px = (right_eye_bot[0]*img_w, right_eye_bot[1]*img_h)
    r_eye_left_px = (right_eye_left[0]*img_w, right_eye_left[1]*img_h)
    r_eye_right_px = (right_eye_right[0]*img_w, right_eye_right[1]*img_h)

    l_eye_h = dist(l_eye_top_px, l_eye_bot_px)
    l_eye_w = dist(l_eye_left_px, l_eye_right_px)
    r_eye_h = dist(r_eye_top_px, r_eye_bot_px)
    r_eye_w = dist(r_eye_left_px, r_eye_right_px)

    l_eye_ratio = l_eye_h / (l_eye_w + 1e-6)
    r_eye_ratio = r_eye_h / (r_eye_w + 1e-6)
    eye_open_avg = (l_eye_ratio + r_eye_ratio) / 2.0

    # eyebrow height (distance brow to eye top)
    left_brow_px = (left_brow[0]*img_w, left_brow[1]*img_h)
    right_brow_px = (right_brow[0]*img_w, right_brow[1]*img_h)
    left_brow_dist = (left_brow_px[1] - l_eye_top_px[1]) / (face_h + 1e-6)
    right_brow_dist = (right_brow_px[1] - r_eye_top_px[1]) / (face_h + 1e-6)
    brow_avg = (left_brow_dist + right_brow_dist) / 2.0

    # Heuristic thresholds (可根据实际情况微调)
    # surprise: mouth open明显 && 眼睛睁大
    if mouth_open_ratio > 0.15 and eye_open_avg > 0.20:
        return 'surprised'
    # happy: smile_ratio 较大（嘴宽/嘴高），且 mouth_open_ratio 不大
    if smile_ratio > 3.0 and 0.10 < mouth_open_ratio < 0.15:
        return 'happy'
    # angry: 眉距低（眉更靠近眼）、且眼睛较眯
    if brow_avg < 0.15 and eye_open_avg < 0.30:
        return 'angry'
    # sad: 眉低但嘴不开且 mouth_open_ratio 很小
    if brow_avg < 0.10 and mouth_open_ratio < 0.01:
        return 'sad'
    # else neutral
    return 'neutral'

# ---- 主程序 ----
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头 (0)。请检查摄像头或索引。")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # gather landmarks normalized
                    lm = [(p.x, p.y) for p in face_landmarks.landmark]
                    # face bbox
                    xs = [p[0] for p in lm]
                    ys = [p[1] for p in lm]
                    x_min = int(min(xs)*img_w)
                    x_max = int(max(xs)*img_w)
                    y_min = int(min(ys)*img_h)
                    y_max = int(max(ys)*img_h)
                    face_cx = int((x_min + x_max) / 2)
                    face_cy = int((y_min + y_max) / 2)
                    face_w = x_max - x_min
                    face_h = max(1, y_max - y_min)

                    # estimate expression
                    expr = estimate_expression(lm, img_w, img_h)

                    # draw bbox and label
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (100, 255, 100), 1)
                    cv2.putText(frame, expr, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # decide emoji position and size (above the face center)
                    emoji_size = int(max(face_w, face_h) * 0.8)
                    emoji_center = (face_cx, max(emoji_size//2 + 5, y_min - emoji_size//2 - 10))

                    # draw emoji onto frame (we draw directly)
                    draw_emoji(frame, emoji_center, emoji_size, expr)

            cv2.imshow('Face Expression -> Emoji (press q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    main()
