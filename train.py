import torch
import torch.nn as nn

from model import CatDogCNN
from data_loader import train_loader


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


    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )

torch.save(model.state_dict(), "cat_dog_model.pth")

print("Model saved successfully!")