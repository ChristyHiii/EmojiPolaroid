# 📸 **Face Polaroid — The Dislocated Emoji Camera**

> A playful fusion of **AI facial expression recognition** and **artistic visual design**.
> Watch your emotions transform into a quirky Polaroid collage — in real time.

---

## 🌈 Overview

**Face Polaroid** is an interactive Python project that blends **computer vision** with **artistic expression**.
Using a webcam, it detects and classifies your facial expression, displaying the corresponding **emoji** floating above your head.

At the same time, it extracts your **facial features** — eyes, nose, and mouth — and reattaches them as **mini Polaroid photos**, slightly misplaced on your face for a surreal yet humorous effect.

When multiple people appear on screen, their facial features are **randomly swapped**, turning every session into a spontaneous visual remix.
You can also **add custom text** to the bottom of the main Polaroid frame and **save the image**.

---

## 🧠 Features

* 🎭 **Real-time Emotion Recognition** – Detects your facial expression and adds a matching emoji above your head.
* 🧩 **Facial Feature Extraction & Misplacement** – Eyes, nose, and mouth appear as separate Polaroid cards repositioned on your face.
* 🔀 **Multi-person Swapping Mode** – Randomly exchanges facial parts between multiple users on camera.
* ✍️ **Custom Polaroid Caption** – Type any message to appear on the Polaroid bottom border.
* 💾 **One-click Save** – Capture and save your current Polaroid composition.
* 🎞️ **Polaroid-style Interface** – Nostalgic instant-camera frame with a modern AI twist.

---

## 🛠️ Tech Stack

| Module                                                                                                       | Purpose                                                          |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Python 3.x**                                                                                               | Main programming language                                        |
| **OpenCV**                                                                                                   | Video capture and image processing                               |
| **Mediapipe / Dlib**                                                                                         | Facial landmark detection                                        |
| **[Facial Expression Recognition Model](https://github.com/rondinellimorais/facial-expression-recognition)** | Pretrained emotion classification model (by *Rondinelli Morais*) |
| **Pillow / Emoji**                                                                                           | Image and emoji rendering                                        |
| **NumPy**                                                                                                    | Matrix and array operations                                      |
| **Tkinter / PyQt (optional)**                                                                                | GUI and Polaroid-style interface                                 |

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe pillow emoji numpy torch torchvision
```

### 2️⃣ Download the Emotion Recognition Model

This project uses the pretrained model from
👉 [rondinellimorais/facial-expression-recognition](https://github.com/rondinellimorais/facial-expression-recognition)

Download the weight file (e.g. `model_best_acc.pth.tar`) and place it in:

```
/models/
```

### 3️⃣ Run the App

```bash
python face_polaroid.py
```

### 4️⃣ How to Use

* The webcam starts automatically.
* Your expression will be recognized and displayed with an emoji.
* Type your custom Polaroid caption in the text box.
* Press **S** to save the current Polaroid snapshot.
* When multiple people appear, their facial features may be randomly swapped — have fun experimenting!

---

## 📷 Example Results

(Insert screenshots or GIFs here)

```
👁️👃👄  Dislocated facial collage  
📸  Real-time emoji overlay  
🪞  Polaroid aesthetic interface
```

---

## 💡 Inspiration

The project is inspired by:

* The nostalgic charm of **instant Polaroid photography**
* The playful tension between **AI perception** and **human identity**
* The question: *“If our features are rearranged or swapped, do we still recognize ourselves?”*

It sits at the intersection of **technology, art, and self-representation**.

---

## 🧑‍💻 Author

**CandyBrown Huang**

> Explorer of AI × Art × Interactive Media
> 📬 [Add your email or portfolio link here, if you’d like]

---

## 🪄 Acknowledgements

This project incorporates the following open-source resources:

* **Facial Expression Recognition Model**
  [rondinellimorais/facial-expression-recognition](https://github.com/rondinellimorais/facial-expression-recognition)
  Author: Rondinelli Morais — *MIT License*

* **Libraries:** OpenCV, Mediapipe, Pillow, Emoji, NumPy

---

## 📜 License

MIT License © 2025 ChrisTing Huang. Feel free to use for learning, research, and creative projects. Contributions and forks are welcome! — just credit the original author🍬

🌟 If you enjoy this project, please ⭐ star the repository and share it.

---

## ✨ Author

**ChrisTing Huang** – [GitHub](https://github.com/ChristyHiii) | [Email](candybrownhuang@gmail.com)
