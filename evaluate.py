import torch

from model import CatDogCNN
from data_loader import test_loader


# Create the model
model = CatDogCNN()


# Load the trained model
model.load_state_dict(torch.load("cat_dog_model.pth"))


# Evaluation mode
model.eval()


correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total


print(f"Test Accuracy: {accuracy:.2f}%")