import os
import random
import shutil


# Original dataset location
SOURCE_DIR = r"T:/pytorch/kagglecatsanddogs_5340/PetImages"

# Our project dataset location
DEST_DIR = "dataset"

# Number of images we want
TRAIN_IMAGES = 2000
TEST_IMAGES = 500


# Make the result reproducible
random.seed(42)


for category in ["Cat", "Dog"]:

    # Original images
    source_folder = os.path.join(SOURCE_DIR, category)

    # Our destination folders
    train_folder = os.path.join(
        DEST_DIR, "train", category.lower() + "s"
    )

    test_folder = os.path.join(
        DEST_DIR, "test", category.lower() + "s"
    )

    # Get image files
    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Shuffle images randomly
    random.shuffle(images)

    # Select required images
    train_images = images[:TRAIN_IMAGES]
    test_images = images[TRAIN_IMAGES:TRAIN_IMAGES + TEST_IMAGES]

    # Copy training images
    for image in train_images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(train_folder, image)

        shutil.copy2(source_path, destination_path)

    # Copy testing images
    for image in test_images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(test_folder, image)

        shutil.copy2(source_path, destination_path)

    print(f"{category}:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Testing images: {len(test_images)}")