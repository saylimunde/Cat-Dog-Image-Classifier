import torch

from data_loader import train_loader
from model import CatDogCNN

# Get one batch of images
images, labels = next(iter(train_loader))

print("Input shape:", images.shape)

# Create the model
model = CatDogCNN()

# Send images through the model
outputs = model(images)


print("Output shape:", outputs.shape)
print("Outputs:", outputs)