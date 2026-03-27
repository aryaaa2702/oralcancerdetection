import os
import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("oral_cancer_model.pth", map_location="cpu"))
model.eval()

target_layers = [model.layer4[-1]]

# GradCAM object
cam = GradCAM(model=model, target_layers=target_layers)

# Folder containing test cancer images
image_folder = "Data/test/Normal"

images = os.listdir(image_folder)

for img_name in images[:5]:   # generate GradCAM for first 5 images

    image_path = os.path.join(image_folder, img_name)
    print("Processing:", image_path)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (96,96))

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img) / 255.0

    input_tensor = torch.tensor(rgb_img).permute(2,0,1).unsqueeze(0)

    targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # enlarge image for better visibility
    visualization_large = cv2.resize(visualization, (600,600))

    # save result
    save_path = f"gradcam_{img_name}.png"
    cv2.imwrite(save_path, visualization_large)

    # show result
    cv2.imshow("GradCAM", visualization_large)
    cv2.waitKey(0)

cv2.destroyAllWindows()