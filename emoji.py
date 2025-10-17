from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf

face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')
camera = cv2.VideoCapture(0)
camera_h = 480
camera_w = 800
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_h)
settings = {'scaleFactor': 1.3, 'minNeighbors': 6, 'minSize': (50, 50)}

labels = ['surprised', 'neutral', 'angry', 'happy', 'sad']
emoji_dict = {
	'surprised': 'emojis/surprised.png',
	'neutral': 'emojis/neutral.png',
	'angry': 'emojis/angry.png',
	'happy': 'emojis/happy.png',
	'sad': 'emojis/sad.png'
}

model = tf.keras.models.load_model('network-5Labels.h5')

# 拍立得边框参数
border_thickness_w = 30
border_thickness_h = 35
border_bottom_extra = 100  # 下边框更宽
border_h = camera_h + border_thickness_h + border_bottom_extra
border_w = camera_w + border_thickness_w * 2

# 拍立得文字
text_input = " "
text_started = False
placeholder = "Enter the text to display on the Polaroid frame"

def overlay_transparent(background, overlay, x, y, scale=1):
	h, w = overlay.shape[0], overlay.shape[1]
	overlay = cv2.resize(overlay, (int(w*scale), int(h*scale)))
	if x >= background.shape[1] or y >= background.shape[0]:
		return background
	h, w = overlay.shape[0], overlay.shape[1]
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
	background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
	return background

while True:
	ret, img = camera.read()
	if not ret:
		break

	img = cv2.flip(img, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detected = face_detection.detectMultiScale(gray, **settings)

	for x, y, w, h in detected:
		# cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
		face = gray[y+5:y+h-5, x+20:x+w-20]
		face = cv2.resize(face, (48,48))
		face = face/255.0

		predictions = model.predict(np.array([face.reshape((48,48,1))]), verbose=0).argmax()
		state = labels[predictions]

		emoji_path = emoji_dict.get(state)
		if emoji_path:
			emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
			if emoji is not None:
				emoji_size = int(w * 0.6)
				img = overlay_transparent(
					img, emoji, x + w//5, max(y - emoji_size, 0),
					scale=emoji_size / emoji.shape[1]
				)

	# 将摄像头画面
	frame_resized = cv2.resize(img, (camera_w, camera_h))

	# 创建白色边框背景
	polaroid = np.ones((border_h, border_w, 3), dtype=np.uint8) * 255
	# 把图像贴入中间
	y_offset = border_thickness_h
	x_offset = border_thickness_w
	polaroid[y_offset:y_offset+camera_h, x_offset:x_offset+camera_w] = frame_resized

	# 拍立得下方说明文字
	display_text = placeholder if not text_started else text_input
	colour = (150, 150, 150) if not text_started else (0, 0, 0)

	(text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
	cv2.putText(polaroid, display_text, (border_w - text_width - 20, border_h - 40),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

	# 固定窗口大小，防止自动缩放导致白边丢失
	cv2.namedWindow('Facial Expression (Polaroid Mirror)', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Facial Expression (Polaroid Mirror)', border_w, border_h)

	# 显示最终结果
	cv2.imshow('Facial Expression (Polaroid Mirror)', polaroid)

	# 键盘输入处理
	key = cv2.waitKey(5) & 0xFF
	if key == 27:	 # ESC to exit
		break
	elif key == 8:	 # Backspace to delete
		if text_started:
			text_input = text_input[:-1]
	elif key == 13:		 # Enter to save image
		filename = datetime.now().strftime("polaroid_%Y%m%d_%H%M%S.jpg")
		cv2.imwrite(filename, polaroid)
		print(f"Saved: {filename}")
	elif 32 <= key <= 126:
		if not text_started:
			text_input = ""
			text_started = True
		text_input += chr(key)

camera.release()
cv2.destroyAllWindows()
