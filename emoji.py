from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf
import random

# === 模型与分类器 ===
face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# === 摄像头设置 ===
camera = cv2.VideoCapture(0)
camera_h, camera_w = 480, 800
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_h)
settings = {'scaleFactor': 1.3, 'minNeighbors': 6, 'minSize': (50, 50)}

# === 模型与标签 ===
labels = ['surprised', 'neutral', 'angry', 'happy', 'sad']
emoji_dict = {
    'surprised': 'emojis/surprised.png',
    'neutral': 'emojis/neutral.png',
    'angry': 'emojis/angry.png',
    'happy': 'emojis/happy.png',
    'sad': 'emojis/sad.png'
}
model = tf.keras.models.load_model('network-5Labels.h5')

# === PNG 图标 ===
icon_eye_L = cv2.imread('icons/eye_L.png', cv2.IMREAD_UNCHANGED)
icon_eye_R = cv2.imread('icons/eye_R.png', cv2.IMREAD_UNCHANGED)
icon_nose = cv2.imread('icons/nose.png', cv2.IMREAD_UNCHANGED)
icon_mouth = cv2.imread('icons/mouth.png', cv2.IMREAD_UNCHANGED)

# === 小拍立得参数 ===
thumb_size = 60
border_thick = 5
border_bottom_extra = 10

# === 拍立得边框参数 ===
border_thickness_w = 30
border_thickness_h = 35
border_bottom_extra_main = 100
border_h = camera_h + border_thickness_h + border_bottom_extra_main
border_w = camera_w + border_thickness_w * 2

# === 全局状态记录 ===
last_face_ids = set()          # 上一帧检测到的 face_id
last_parts_available = {}        # 保存互换后的结果
swap_map = {
    'left_eye': {},
    'right_eye': {},
    'nose': {},
    'mouth': {}
}

# === 文字输入 ===
text_input = " "
text_started = False
placeholder = "Enter the text to display on the Polaroid frame"

# === 稳定跟踪 ===
face_positions = {}   # {face_id: (x, y, w, h)}
face_offsets = {}     # {face_id: (offset_x, offset_y)}
next_face_id = 0

def overlay_transparent(background, overlay, x, y, scale=1):
    """贴图（支持透明通道）"""
    if overlay is None:
        return background
    h, w = overlay.shape[:2]
    overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)))
    if x >= background.shape[1] or y >= background.shape[0]:
        return background
    h, w = overlay.shape[:2]
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]
    if overlay.shape[2] < 4:
        return background
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_img
    return background

def get_face_id(x, y, w, h):
    """根据脸位置生成稳定 face_id"""
    global next_face_id
    for fid, (fx, fy, fw, fh) in face_positions.items():
        if abs(x - fx) < 60 and abs(y - fy) < 60:
            face_positions[fid] = (x, y, w, h)
            return fid
    fid = next_face_id
    next_face_id += 1
    face_positions[fid] = (x, y, w, h)
    return fid

show_polaroids = True  # 是否显示五官拍立得
def mouse_callback(event, x, y, flags, param):
    global show_polaroids
    if event == cv2.EVENT_RBUTTONDOWN:
        show_polaroids = not show_polaroids
        print("Show polaroids:", show_polaroids)


while True:
    ret, img = camera.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = face_detection.detectMultiScale(gray, **settings)

    face_parts = {}
    for (x, y, w, h) in detected:
        fid = get_face_id(x, y, w, h)

        # 分配固定偏移
        if fid not in face_offsets:
            face_offsets[fid] = (random.randint(-10, 10), random.randint(-10, 10))
        offset_x, offset_y = face_offsets[fid]

        # === 各部位检测 ===
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        left_eye_img, right_eye_img, nose_img, mouth_img = None, None, None, None

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 10, minSize=(30, 30))
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_center_x = ex + ew / 2
            if eye_center_x < w / 2:
                left_eye_img = roi_color[ey:ey + eh, ex:ex + ew]
            else:
                right_eye_img = roi_color[ey:ey + eh, ex:ex + ew]

        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(30, 30))
        if len(noses) > 0:
            nx, ny, nw, nh = noses[0]
            nose_img = roi_color[ny:ny + nh, nx:nx + nw]

        lower_face = roi_gray[int(h * 0.6):, :]
        mouths = mouth_cascade.detectMultiScale(lower_face, 1.3, 8, minSize=(40, 40))
        if len(mouths) > 0:
            mx, my, mw, mh = mouths[0]
            my += int(h * 0.6)
            mouth_img = roi_color[my:my + mh, mx:mx + mw]

        scale = w / 220.0       # 缩放系数
        face_parts[fid] = {
            "x": x, "y": y, "w": w, "h": h,
            "scale": scale,
            "left_eye": left_eye_img,
            "right_eye": right_eye_img,
            "nose": nose_img,
            "mouth": mouth_img,
        }

    current_face_ids = set(face_parts.keys())
    trigger_shuffle = False

    # === 检测到人脸变化时触发互换 ===
    if current_face_ids != last_face_ids:
        trigger_shuffle = True

    for fid, parts in face_parts.items():
        if fid not in last_parts_available:
            trigger_shuffle = True
            break
        for pname in ['left_eye', 'right_eye', 'nose', 'mouth']:
            prev_has = last_parts_available[fid].get(pname) is not None
            now_has = parts[pname] is not None
            if not prev_has and now_has:
                trigger_shuffle = True
                break

    # === 互换五官 ===
    if trigger_shuffle and len(face_parts) > 1:
        part_names = ["left_eye", "right_eye", "nose", "mouth"]
        for part in part_names:
            parts_available = [fid for fid in face_parts if face_parts[fid][part] is not None]
            if len(parts_available) > 1:
                shuffled = parts_available.copy()
                random.shuffle(shuffled)

                # 防止每个人拿回自己
                if all(a == b for a, b in zip(parts_available, shuffled)):
                    random.shuffle(shuffled)

                for src, dst in zip(parts_available, shuffled):
                    swap_map[part][dst] = src  # “dst 用 src 的五官”
        print("🔁 五官互换完成！")

    # === 更新状态 ===
    last_face_ids = current_face_ids
    last_parts_available = {
        fid: {p: (face_parts[fid][p] if face_parts[fid][p] is not None else None)
            for p in ["left_eye", "right_eye", "nose", "mouth"]}
        for fid in face_parts
    }

    # === 绘制每张脸 ===
    for fid, fdata in face_parts.items():
        x, y, w, h = fdata["x"], fdata["y"], fdata["w"], fdata["h"]
        scale = fdata["scale"]
        thumb_size_scaled = int(thumb_size * scale)
        border_thick_scaled = int(border_thick * scale)
        border_bottom_extra_scaled = int(border_bottom_extra * scale)
        offset_x, offset_y = face_offsets[fid]

        # === 表情识别 ===
        face_gray = gray[y + 5:y + h - 5, x + 20:x + w - 20]
        face_gray = cv2.resize(face_gray, (48, 48))
        face_gray = face_gray / 255.0
        predictions = model.predict(np.array([face_gray.reshape((48, 48, 1))]), verbose=0).argmax()
        state = labels[predictions]

        # === 部位拍立得固定位置 ===
        positions = [
            ("left_eye", fdata["left_eye"], icon_eye_L, (x + int(w * 0.26) - thumb_size_scaled // 2, y + int(h * 0.25) - thumb_size_scaled // 2)),
            ("right_eye", fdata["right_eye"], icon_eye_R, (x + int(w * 0.68) - thumb_size_scaled // 2, y + int(h * 0.32) - thumb_size_scaled // 2)),
            ("nose", fdata["nose"], icon_nose, (x + int(w * 0.52) - thumb_size_scaled // 2, y + int(h * 0.55) - thumb_size_scaled // 2)),
            ("mouth", fdata["mouth"], icon_mouth, (x + int(w * 0.44) - thumb_size_scaled // 2, y + int(h * 0.81) - thumb_size_scaled // 2))
        ]

        if show_polaroids:
            for name, part_img, icon_img, (px, py) in positions:
                dx, dy = face_offsets[fid]
                px += dx
                py += dy
                card_w = thumb_size_scaled + border_thick_scaled * 2
                card_h = thumb_size_scaled + border_thick_scaled * 2 + border_bottom_extra_scaled

                if fid in swap_map[name]:
                    src_fid = swap_map[name][fid]
                    if src_fid in face_parts and face_parts[src_fid][name] is not None:
                        part_img = face_parts[src_fid][name]

                # === 位置边界检查 ===
                if px < 0 or py < 0 or px + card_w > img.shape[1] or py + card_h > img.shape[0]:
                    continue

                overlay = img.copy()
                cv2.rectangle(overlay, (px + 2, py + 2), (px + card_w + 2, py + card_h + 2), (180, 180, 180), -1)
                cv2.rectangle(overlay, (px, py), (px + card_w, py + card_h), (255, 255, 255), -1)
                img = cv2.addWeighted(overlay, 0.9, img, 0.1, 0)

                img_region = (px + border_thick_scaled, py + border_thick_scaled)
                if part_img is not None:
                    resized = cv2.resize(part_img, (thumb_size_scaled, thumb_size_scaled))
                    img[img_region[1]:img_region[1] + thumb_size_scaled,
                        img_region[0]:img_region[0] + thumb_size_scaled] = resized

                # 在底部贴 icon 图片（代替文字）
                if icon_img is not None:
                    icon_scale = 0.35 * scale
                    icon_h, icon_w = int(icon_img.shape[0] * icon_scale), int(icon_img.shape[1] * icon_scale)
                    if name in ['left_eye', 'mouth']:
                        text_x = int(px + border_thick)
                    else:
                        text_x = int(px + card_w - icon_w - 5)
                    text_y = int(py + card_h - icon_h - 5)  # 保持底边距
                    img = overlay_transparent(
                        img, icon_img, text_x, text_y, scale=icon_scale
                    )

        # === 表情 emoji  ===
        emoji_path = emoji_dict.get(state)
        if emoji_path:
            emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
            if emoji is not None:
                img = overlay_transparent(img, emoji, x + w // 5, max(y - int(h * 0.80), 0), scale=w * 0.65 / emoji.shape[1])

    # === 主拍立得边框 ===
    frame_resized = cv2.resize(img, (camera_w, camera_h))
    polaroid = np.ones((border_h, border_w, 3), dtype=np.uint8) * 255
    polaroid[border_thickness_h:border_thickness_h + camera_h,
            border_thickness_w:border_thickness_w + camera_w] = frame_resized

    display_text = placeholder if not text_started else text_input
    colour = (150, 150, 150) if not text_started else (0, 0, 0)
    (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(polaroid, display_text, (border_w - text_width - 20, border_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    cv2.namedWindow('Facial Expression (Polaroid Mirror)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Facial Expression (Polaroid Mirror)', mouse_callback)
    cv2.resizeWindow('Facial Expression (Polaroid Mirror)', border_w, border_h)
    cv2.imshow('Facial Expression (Polaroid Mirror)', polaroid)

    # === 键盘事件 ===
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC 退出
        break
    elif key == 8:
        if text_started:
            text_input = text_input[:-1]
    elif key == 13:  # Enter 保存图片
        filename = datetime.now().strftime("polaroid_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(filename, polaroid)
        print(f"Saved: {filename}")
    elif 32 <= key <= 126:
        if not text_started:
            text_input = " "
            text_started = True
        text_input += chr(key)

camera.release()
cv2.destroyAllWindows()
