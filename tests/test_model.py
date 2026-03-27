import torch
from model import get_model
from train_setup import train_loader

model =get_model()

images, labels=next(iter(train_loader))

outputs=model(images)

print("Input batch shape: ", images.shape)
print("Output shape: ", outputs.shape)