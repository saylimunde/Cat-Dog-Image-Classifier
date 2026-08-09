# 🐱🐶 Cat vs Dog Image Classifier

A simple image classification project built with **PyTorch** and a Convolutional Neural Network (CNN).

The model is trained to classify images into two classes:

- 🐱 Cats
- 🐶 Dogs

## 📌 Project Overview

This project demonstrates the complete workflow of building an image classification model using PyTorch.

The workflow includes:

1. Loading and organizing image data
2. Image preprocessing
3. Creating a DataLoader
4. Building a CNN
5. Training the model
6. Evaluating test accuracy
7. Saving the trained model
8. Predicting new images

## 📂 Project Structure

```text
cat_dog_classifier/
│
├── images/
│   ├── my_cat.jpg
│   └── my_dog.jpg
│
├── model.py
├── data_loader.py
├── train.py
├── evaluate.py
├── predict.py
├── check_images.py
├── prepare_dataset.py
├── test_model.py
├── cat_dog_model.pth
├── requirements.txt
├── .gitignore
└── README.md