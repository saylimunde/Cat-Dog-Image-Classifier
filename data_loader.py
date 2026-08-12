import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


# -----------------------------
# Training transform
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -----------------------------
# Validation transform
# -----------------------------

validation_transform = transforms.Compose([
    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -----------------------------
# Load training dataset
# -----------------------------

train_dataset_full = ImageFolder(
    root="dataset/train",
    transform=train_transform
)


# -----------------------------
# Load validation dataset
# -----------------------------

validation_dataset_full = ImageFolder(
    root="dataset/train",
    transform=validation_transform
)


# -----------------------------
# Train / validation split
# -----------------------------

train_size = int(0.8 * len(train_dataset_full))

validation_size = (
    len(train_dataset_full) - train_size
)


generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(train_dataset_full),
    generator=generator
).tolist()


train_indices = indices[:train_size]

validation_indices = indices[train_size:]


train_dataset = Subset(
    train_dataset_full,
    train_indices
)


validation_dataset = Subset(
    validation_dataset_full,
    validation_indices
)


# -----------------------------
# DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)


# -----------------------------
# Test transform
# -----------------------------

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -----------------------------
# Test dataset
# -----------------------------

test_dataset = ImageFolder(
    root="dataset/test",
    transform=test_transform
)


# -----------------------------
# Test DataLoader
# -----------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)