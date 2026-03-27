from dataset_loader import OralCancerDataset

dataset=OralCancerDataset("data/train")

print("Total Samples:", len(dataset))

img, label=dataset[0]
print("Image tensor shape:", img.shape)
print("Label:", label)