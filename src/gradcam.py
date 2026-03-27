import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "Data/test/Normal/00c9f38dd043d3005774ee6e0503efaafeee993b.tif"   # <-- change this to your image path
MODEL_PATH = "best_oral_cancer_model.pth"
OUTPUT_PATH = "gradcam_result.jpg"

CLASS_NAMES = ["Normal", "OSCC"]   # 0 = Normal, 1 = OSCC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
    gradients.append(grad_output[0])

def save_activation(module, input, output):
    activations.append(output)

# Register hooks on last conv layer
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
img = cv2.resize(original_img, (224, 224))
img = img.astype("float32") / 255.0

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

for i in range(3):
    img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

input_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

# -----------------------------
# FORWARD PASS
# -----------------------------
output = model(input_tensor)
pred_class = torch.argmax(output, dim=1).item()
pred_score = torch.softmax(output, dim=1)[0][pred_class].item()

print("Predicted Class:", CLASS_NAMES[pred_class])
print("Confidence:", round(pred_score * 100, 2), "%")

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
cam = cam / cam.max()

# Resize CAM to original image size
cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))

# Convert to heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Overlay heatmap on original image
overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

# -----------------------------
# SAVE RESULT
# -----------------------------
cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"Grad-CAM result saved as: {OUTPUT_PATH}")

# -----------------------------
# DISPLAY RESULT
# -----------------------------
plt.figure(figsize=(12, 4))

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
plt.title(f"Grad-CAM: {CLASS_NAMES[pred_class]}")
plt.axis("off")

plt.tight_layout()
plt.show()