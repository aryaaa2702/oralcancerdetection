import os
import cv2
import torch
from torch.utils.data import Dataset

class OralCancerDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []

        oscc_dir = os.path.join(root_dir, "OSCC")
        normal_dir = os.path.join(root_dir, "Normal")

        # Load OSCC images (label = 1)
        for img_name in os.listdir(oscc_dir):
            img_path = os.path.join(oscc_dir, img_name)
            if os.path.isfile(img_path):
                self.images.append(img_path)
                self.labels.append(1)

        # Load Normal images (label = 0)
        for img_name in os.listdir(normal_dir):
            img_path = os.path.join(normal_dir, img_name)
            if os.path.isfile(img_path):
                self.images.append(img_path)
                self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.images[idx])

            if img is None:
                raise ValueError("Image not loaded")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            # Convert to float and scale to [0,1]
            img = img.astype("float32") / 255.0

            # Normalize for pretrained ResNet
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            for i in range(3):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

            # Convert to tensor and rearrange dimensions
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

            # Label
            label = torch.tensor(self.labels[idx], dtype=torch.long)

            return img, label

        except Exception:
            print(f"⚠️ Skipping bad image: {self.images[idx]}")
            return self.__getitem__((idx + 1) % len(self))