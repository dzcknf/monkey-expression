# 🐒 Monkey Detection & Expression Recognition
### Real-Time Facial Expression & Gesture Recognition using Deep Learning & Computer Vision

This project is a real-time **facial expression and gesture recognition system** built from scratch using **PyTorch**, **OpenCV**, and **Python**. 
It classifies expressions captured from a webcam and displays a matching monkey meme or response on screen in real-time.

---

## ✋ Supported Gestures & Expressions (5 Classes)
The model is trained to recognize 5 specific classes:
1. **Index Finger Pointing Up** (*Menunjuk jari telunjuk ke atas*)
2. **Index Finger on Lips** (*Memegang bibir dengan jari telunjuk*)
3. **Surprised Pose** (*Pose kaget*)
4. **Natural Expression** (*Natural*)
5. **Viral "Kicau Kicau Mania" Pose** (*Pose kicau kicau mania yang viral*)

---

## 🧠 Core Concepts
I trained a **Convolutional Neural Network (CNN)** from scratch to recognize these 5 facial expressions and gestures from webcam images. 
Using **OpenCV**, the program streams video frames in real-time, preprocesses them, feeds them into the trained PyTorch model, and visualizes the predictions side-by-side.

---

## ⚙️ Tech Stack 

### 🧩 **Python**
The base language for the entire project.

### 🔥 **PyTorch**
Used to:
- Build a fully custom **CNN model** (no pre-trained weights).
- Handle tensor operations, forward propagation, and classification.
- Train and evaluate the model efficiently.

### 🎥 **OpenCV**
Used to:
- Capture frames from the webcam in real-time.
- Display live video feeds and trigger images in a single window.
- Handle color space conversions and image resizing.

### 🧠 **NumPy & Pillow (PIL)**
Used for numeric array operations, image resizing, and preparing frames for PyTorch tensors.

---

## 🚀 How to Run the Project

### 1. Setup Virtual Environment & Activate It
Open your terminal (PowerShell) in VS Code and run:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

### Install Dependencies 
python3 -m pip install --upgrade pip --break-system-packages
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
python3 -m pip install opencv-python pillow numpy

### Capture Training Data
python3 capture_dataset.py

### Train the Model
python3 train_model.py

### Run the program live with webcam
python3 detect_expression.py

### Exit
press esc for exit

### for delete the photos database
Remove-Item -Path "images\*\*.jpg" -Force



sdc
