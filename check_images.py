from PIL import Image
import os


folders = [
    "dataset/train/cats",
    "dataset/train/dogs",
    "dataset/test/cats",
    "dataset/test/dogs"
]


bad_images = []


for folder in folders:

    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)

        try:
            with Image.open(file_path) as img:
                img.verify()

        except Exception:
            bad_images.append(file_path)


print("Bad images found:", len(bad_images))

for image in bad_images:
    print(image)