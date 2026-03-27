import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "Data/test/Normal/0a089a1cd87d850be5e7a65a97764ad3ab320dfd.tif"   # <-- change this image path
MODEL_PATH = "best_oral_cancer_model.pth"
OUTPUT_PATH = "gradcam_demo_result.jpg"

CLASS_NAMES = ["Normal", "OSCC"]   # 0 = Normal, 1 = OSCC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# DETERMINE ACTUAL LABEL FROM FOLDER
# -----------------------------
if "OSCC" in IMAGE_PATH:
    actual_label = 1
elif "Normal" in IMAGE_PATH:
    actual_label = 0
else:
    actual_label = None

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# HOOKS FOR GRAD-CAM
# -----------------------------
gradients = []
activations = []

def save_gradient(module, grad_input, grad_output):
    gradients.clear()
    gradients.append(grad_output[0])

def save_activation(module, input, output):
    activations.clear()
    activations.append(output)

# Use last conv layer of ResNet18
target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Image not found: {IMAGE_PATH}")

original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(original_img, (224, 224))
img = img_resized.astype("float32") / 255.0

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

for i in range(3):
    img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

input_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

# -----------------------------
# FORWARD PASS
# -----------------------------
output = model(input_tensor)
probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()

pred_class = np.argmax(probs)
pred_confidence = probs[pred_class] * 100
normal_prob = probs[0] * 100
oscc_prob = probs[1] * 100

print("Predicted Class:", CLASS_NAMES[pred_class])
print("Prediction Confidence:", round(pred_confidence, 2), "%")
print("Normal Probability:", round(normal_prob, 2), "%")
print("OSCC Probability:", round(oscc_prob, 2), "%")

if actual_label is not None:
    print("Actual Class:", CLASS_NAMES[actual_label])

# -----------------------------
# BACKWARD PASS
# -----------------------------
model.zero_grad()
output[0, pred_class].backward()

# -----------------------------
# GENERATE GRAD-CAM
# -----------------------------
grads = gradients[0].cpu().data.numpy()[0]
acts = activations[0].cpu().data.numpy()[0]

weights = np.mean(grads, axis=(1, 2))
cam = np.zeros(acts.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)

if cam.max() != 0:
    cam = cam / cam.max()

cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))

# Heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Overlay
overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

# -----------------------------
# SAVE RESULT
# -----------------------------
cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"Grad-CAM result saved as: {OUTPUT_PATH}")

# -----------------------------
# DISPLAY RESULT
# -----------------------------
actual_text = CLASS_NAMES[actual_label] if actual_label is not None else "Unknown"

title_text = (
    f"Predicted: {CLASS_NAMES[pred_class]} ({pred_confidence:.2f}%)\n"
    f"Actual: {actual_text}\n"
    f"Normal: {normal_prob:.2f}% | OSCC: {oscc_prob:.2f}%"
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap)
plt.title("Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(title_text)
plt.axis("off")

plt.tight_layout()
plt.show()