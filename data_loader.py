from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(
    root="dataset/train",
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


test_dataset = ImageFolder(
    root="dataset/test",
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)



print("Classes:", train_dataset.classes)
print("Class to index:", train_dataset.class_to_idx)
print("Number of images:", len(train_dataset))
print("Number of batches:", len(train_loader))

# from PIL import Image

# image_path = r"dataset\train\cats"

# print(train_transform)

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("First image shape:", images[0].shape)
print("First 10 labels:", labels[:10])