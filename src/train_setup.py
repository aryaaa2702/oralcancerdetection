from torch.utils.data import DataLoader
from dataset_loader import OralCancerDataset
from torch.utils.data import Subset

train_dataset=OralCancerDataset("data/train")
val_dataset=OralCancerDataset("data/val")

train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=32, shuffle=False)

images, labels=next(iter(train_loader))

print("Batch image shape: ", images.shape)
print("Batch labels shape:", labels.shape)