# 🐒 Monkey Detection  
### Real-Time Facial Expression Recognition using Deep Learning & Computer Vision

This project is a real-time **facial expression recognition system** built from scratch using **PyTorch**, **OpenCV**, and **Python**.  
It classifies facial expressions captured from a webcam and displays a matching monkey meme on screen — all happening live.
 
---

## 🧠 Core Concepts
I trained a **Convolutional Neural Network (CNN)** to recognize facial expressions and gestures from nearly 500 images of my own face.  
Using **OpenCV**, I stream video frames from the webcam, preprocess them, feed them to the model, and visualize the prediction side-by-side with a trigger image that corresponds to the detected emotion.

---

## ⚙️ Tech Stack 

### 🧩 **Python 3.13**
The base language for the entire project

### 🔥 **PyTorch**
Used to:
- Build a fully custom **CNN model** (no pre-trained weights).
- Handle **tensor operations**, **forward propagation**, and **softmax classification**.
- Train and evaluate the model efficiently using GPU acceleration via Apple’s MPS backend.

### 🎥 **OpenCV**
Used to:
- Capture frames from the webcam in real-time.
- Display live video feeds and trigger images in a single window.
- Handle color space conversions and image resizing.

### 🧠 **NumPy**
Used for:
- Merging the webcam feed and trigger images.
- Efficient numeric operations while handling real-time image arrays.

### 🖼️ **Pillow (PIL)**
Used to:
- Convert OpenCV frames into a format compatible with PyTorch transforms.
- Resize and preprocess images before feeding them to the model.

### 🧰 **Torchvision**
Used for:
- `ImageFolder` — automatically managing labeled training data.
- Data transformations like `Resize()` and `ToTensor()`.
- Streamlining dataset management for the CNN.

---

## 🧠 What I Learned
- How to architect and train a **Convolutional Neural Network** from the ground up.  
- How to use **PyTorch** for model definition, training loops, and GPU acceleration.  
- How to integrate **OpenCV** with deep learning models for real-time applications.  
- How to automate **dataset collection** through webcam capture.  
- The importance of **balanced datasets** and diverse image samples.  
- How to optimize preprocessing pipelines for real-time performance.  
- How to debug and tune models for improved accuracy.

---

## 🚀 How to Run the Project

```bash
python3 -m venv venv
source venv/bin/activate

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

### for delete the photos database
Remove-Item -Path "images\*\*.jpg" -Force

sdc
