import os
import cv2

print("Script started")

DATASET_PATH = "Data/train/OSCC"  

images = os.listdir(DATASET_PATH)
print("Total images found:", len(images))

image_path = os.path.join(DATASET_PATH, images[0])
print("Loading:", image_path)

img = cv2.imread(image_path)

if img is None:
    print("Image not loaded")
else:
    print("Image loaded")
    print("Image shape:", img.shape)