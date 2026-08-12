import torch
from model import CatDogCNN
from data_loader import test_loader


# Create model
model = CatDogCNN()

# Load trained model
model.load_state_dict(torch.load("cat_dog_model.pth"))

# Evaluation mode
model.eval()


# Counters
cat_correct = 0
cat_wrong = 0

dog_correct = 0
dog_wrong = 0


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        for actual, prediction in zip(labels, predicted):

            if actual == 0:  # Cat

                if prediction == 0:
                    cat_correct += 1
                else:
                    cat_wrong += 1

            else:  # Dog

                if prediction == 1:
                    dog_correct += 1
                else:
                    dog_wrong += 1


print("----- Results -----")

print("Cats correctly predicted:", cat_correct)
print("Cats incorrectly predicted as dogs:", cat_wrong)

print("Dogs correctly predicted:", dog_correct)
print("Dogs incorrectly predicted as cats:", dog_wrong)