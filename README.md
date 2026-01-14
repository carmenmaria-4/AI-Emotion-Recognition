# 🎭 Real-Time Emotion Recognition App

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)

An interactive desktop application that detects human faces and classifies emotions in real-time. This project covers the entire ML pipeline, from model fine-tuning to a functional GUI.

---

## ✨ Features

* **Real-Time Monitoring:** Processes live video stream from the webcam instantly.
* **Dynamic Interface:** The GUI (built with Tkinter) changes its background color and displays specific emojis based on the detected emotion.
* **Multi-Class Classification:** Recognizes 7 fundamental emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
* **Model Tuning:** Includes the script used for retraining the CNN to achieve better accuracy.

---

## 🛠️ Project Structure

📂 Emotion_Recognition/
│
├── 📂 models/   
│   ├── fine_tuned_model.hdf5            # Main optimized model
│   ├── _mini_XCEPTION.102-0.66.hdf5     # Base architecture model
│   └── cnn.py                           # Model architecture definition
│
├── 📂 haarcascade_files/                
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml              
│
├── main.py                             # Main GUI Application
├── fine_tune.py                         # Model retraining & tuning script
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation

---

## 📊 Dataset & Training

The model was fine-tuned using the **FER2013** dataset and custom augmented data. Due to storage limits on GitHub, the raw training images and extra experimental models are hosted externally:

🔗 **[Download Training Data & Extra Models]**
https://drive.google.com/drive/folders/1Qrfp_ELt-urvr4xc_3zAdgystlBSSrMh?usp=sharing
---

## 🚀 Installation & Usage

### 1. Prerequisites
* **Python 3.11**
* A working webcam

### 2. Setup
1. **Clone the repository** to your local machine.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
