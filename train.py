import torch
import torch.nn as nn

from model import CatDogCNN
from data_loader import train_loader,validation_loader


# Create the model
model = CatDogCNN()


# Loss function
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Number of training epochs
num_epochs = 5


# Training loop
for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Add loss
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    average_loss = running_loss / len(train_loader)

    training_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}, "
        f"Training Accuracy: {training_accuracy:.2f}%"
    )

    # Validation
    model.eval()

    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            validation_total += labels.size(0)

            validation_correct += (predicted == labels).sum().item()


    validation_accuracy = 100 * validation_correct / validation_total

    print(
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )
torch.save(model.state_dict(), "cat_dog_model.pth")

print("Model saved successfully!")