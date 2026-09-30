import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import imageio

# Model cnn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load data
from torchvision import datasets
dataset = datasets.ImageFolder("images", transform=transforms.ToTensor())
expressions = dataset.classes
num_classes = len(expressions)

# Load model
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load("expression_model.pth", map_location=device))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Resize images for webcam window
display_height = 480
display_width = 640

# Load triggers (JPG + GIF)
trigger_images = {}

for expr in expressions:
    gif_path = f"triggers/{expr}.gif"
    jpg_path = f"triggers/{expr}.jpg"

    if os.path.exists(gif_path):
        gif_frames = imageio.mimread(gif_path)
        processed_frames = []

        for frame in gif_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (display_width, display_height))
            processed_frames.append(frame)

        trigger_images[expr] = processed_frames

    elif os.path.exists(jpg_path):
        img = cv2.imread(jpg_path)
        if img is not None:
            img = cv2.resize(img, (display_width, display_height))
            trigger_images[expr] = img

# Default set to neutral expression
neutral_display = trigger_images.get("neutral", np.zeros((display_height, display_width, 3), dtype=np.uint8))

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, display_width)
cap.set(4, display_height)
gif_frame = 0

# Window spacing so split screen 
cv2.namedWindow("Expression Split View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Expression Split View", 1280, 480)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if not ret:
        break

    # Frame to tensor
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    confidence_value = confidence.item()
    label = expressions[pred.item()] if confidence_value > 0.6 else "neutral"
    trigger_path = f"triggers/{label}"

    # Scrap low confidence predictions
    print(f"Detected: {label} | Confidence: {confidence_value:.2f} | Trigger: {trigger_path}")

    # Text overlay
    cv2.putText(frame, f"{label} ({confidence_value:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Fetch image
    display_data = trigger_images.get(label, neutral_display)

    if isinstance(display_data, list):  # GIF
        display_img = display_data[gif_frame % len(display_data)]
        gif_frame += 1
    else:  # JPG
        display_img = display_data

    # Webcam and image on same window
    if display_img is not None and display_img.shape == frame.shape:
        combined = np.hstack((frame, display_img))
    else:
        combined = frame

    cv2.imshow("Expression Split View", combined)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
