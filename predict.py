from PIL import Image
import torch
from torchvision import transforms

from model import CatDogCNN


# Create the model
model = CatDogCNN()


# Load trained weights
model.load_state_dict(
    torch.load("cat_dog_model.pth")
)


# Put model in evaluation mode
model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# Load image
image = Image.open("T:/pytorch/cat_dog_classifier/images/my_dog.jpg").convert("RGB")


# Apply preprocessing
image = transform(image)


# Add batch dimension
image = image.unsqueeze(0)


# Make prediction
with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    _, predicted = torch.max(output, 1)


classes = ["cats", "dogs"]

print("Cat probability:", probabilities[0][0].item() * 100)
print("Dog probability:", probabilities[0][1].item() * 100)

prediction = classes[predicted.item()]

print("Prediction:", prediction)