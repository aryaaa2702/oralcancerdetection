import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Subset
import random

from dataset_loader import OralCancerDataset


def get_random_subset(dataset, subset_size):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:min(subset_size, len(dataset))])


def main():
    print("STEP 1: Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load datasets
    print("Loading training dataset...")
    full_train_dataset = OralCancerDataset("Data/train")
    train_dataset = get_random_subset(full_train_dataset, 12000)
    print("Training samples:", len(train_dataset))

    print("Loading validation dataset...")
    full_val_dataset = OralCancerDataset("Data/val")
    val_dataset = get_random_subset(full_val_dataset, 3000)
    print("Validation samples:", len(val_dataset))

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    print("DataLoaders ready")

    # Model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 6   # good balance for your laptop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx == 0:
                print("First training batch loaded")

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_oral_cancer_model.pth")
            print("Best model saved!")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()