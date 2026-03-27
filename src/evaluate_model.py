import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from collections import Counter
import random

from dataset_loader import OralCancerDataset


def get_random_subset(dataset, subset_size):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:min(subset_size, len(dataset))])


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load test dataset
print("Loading test dataset...")
full_test_dataset = OralCancerDataset("Data/test")
test_dataset = get_random_subset(full_test_dataset, 1000)   # enough for evaluation
print("Test dataset loaded")
print("Total test samples:", len(test_dataset))

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# Load model
print("Loading model...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("best_oral_cancer_model.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded")

all_preds = []
all_labels = []

print("Evaluating...")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):

        if batch_idx == 0:
            print("First test batch loaded")

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        if batch_idx == 0:
            print("Sample Predictions:", preds[:10].cpu().numpy())
            print("Actual Labels:", labels[:10].cpu().numpy())

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print("\nTest Accuracy:", accuracy)
print("Test Accuracy (%):", accuracy * 100)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))

print("\nTotal samples evaluated:", len(all_preds))