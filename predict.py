import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CatDogCNN


# Load the trained model
model = CatDogCNN()

model.load_state_dict(
    torch.load("cat_dog_model.pth")
)

model.eval()


# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# Change this to the image you want to predict
image_path = "T:/pytorch/cat_dog_classifier/images/my_cat2.jpg"


# Load image
image = Image.open(image_path).convert("RGB")


# Apply transformation
image = transform(image)


# Add batch dimension
image = image.unsqueeze(0)


# Make prediction
with torch.no_grad():

    outputs = model(image)

    probabilities = F.softmax(outputs, dim=1)

    cat_probability = probabilities[0][0].item() * 100
    dog_probability = probabilities[0][1].item() * 100


# Determine prediction
if cat_probability > dog_probability:
    prediction = "Cat"
    confidence = cat_probability
else:
    prediction = "Dog"
    confidence = dog_probability


print("----- Prediction -----")

print(f"Cat probability: {cat_probability:.2f}%")
print(f"Dog probability: {dog_probability:.2f}%")

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")