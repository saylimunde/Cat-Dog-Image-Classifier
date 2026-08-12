# 🐱🐶 Cat vs Dog Image Classifier

A Convolutional Neural Network (CNN) built with PyTorch to classify images as either **cats** or **dogs**.

The project covers the complete machine learning workflow:

* Loading and preparing image data
* Training a CNN
* Validation during training
* Data augmentation
* Dropout for reducing overfitting
* Model evaluation
* Confusion matrix analysis
* Single-image prediction with confidence

---

## 📌 Project Overview

The goal of this project is to build an image classification model that can distinguish between cats and dogs.

The model was built from scratch using **PyTorch** rather than using a pre-trained model.

### Final Test Accuracy

**76.90%**

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* Pillow
* Git & GitHub

---

## 📂 Project Structure

```text
cat_dog_classifier/
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   │
│   └── test/
│       ├── cats/
│       └── dogs/
│
├── model.py
├── data_loader.py
├── train.py
├── evaluate.py
├── confusion_matrix.py
├── predict.py
├── cat_dog_model.pth
├── requirements.txt
├── README.md
└── images/
```

---

## 🧠 CNN Architecture

The model contains three convolutional blocks.

### Block 1

```text
Conv2D: 3 → 16 channels
ReLU
MaxPool
```

### Block 2

```text
Conv2D: 16 → 32 channels
ReLU
MaxPool
```

### Block 3

```text
Conv2D: 32 → 64 channels
ReLU
MaxPool
```

After the convolution layers:

```text
Flatten
↓
Linear: 16384 → 128
↓
ReLU
↓
Dropout(0.5)
↓
Linear: 128 → 2
```

The final two outputs represent:

```text
0 → Cat
1 → Dog
```

---

## 🔄 Data Preprocessing

Images are resized to:

```text
128 × 128 pixels
```

Training images use:

```text
RandomHorizontalFlip
```

to provide some variation during training.

Images are converted to tensors and normalized using:

```text
Mean = [0.5, 0.5, 0.5]
Standard deviation = [0.5, 0.5, 0.5]
```

The training dataset contains **3,999 images**.

It is divided into:

```text
Training   → 3,199 images
Validation →   800 images
```

The test dataset contains:

```text
Test → 1,000 images
```

Validation and test images are not randomly augmented.

---

## ⚙️ Training

The model was trained using:

### Loss Function

```python
CrossEntropyLoss
```

### Optimizer

```python
Adam
```

### Learning Rate

```text
0.001
```

### Batch Size

```text
32
```

### Epochs

```text
5
```

---

## 📊 Training Results

Final training results:

| Metric              |     Result |
| ------------------- | ---------: |
| Training Accuracy   |     78.09% |
| Validation Accuracy |     76.25% |
| Test Accuracy       | **76.90%** |

---

## 🧪 Model Evaluation

The final model achieved:

### Test Accuracy

**76.90%**

The model was also evaluated separately for cats and dogs.

| Class | Correct | Incorrect | Accuracy |
| ----- | ------: | --------: | -------: |
| Cats  |     393 |       107 |    78.6% |
| Dogs  |     376 |       124 |    75.2% |

### Confusion Matrix

```text
                    Predicted
                  Cat       Dog
Actual Cat       393       107
Actual Dog       124       376
```

The model performs relatively similarly on both classes, with slightly better performance on cats.

---

## 🔬 Experiments

Several experiments were performed during development.

### Baseline CNN

Test Accuracy:

**73.40%**

### Dropout

Dropout was added after the first fully connected layer:

```python
nn.Dropout(0.5)
```

This helped reduce the gap between training and validation performance.

### Separate Validation Transform

Training images use augmentation while validation images are evaluated without random augmentation.

This produced a small improvement in test performance.

### Batch Normalization

Batch Normalization was tested but reduced performance in this particular setup.

Therefore, it was removed from the final model.

### Final Model

The final model achieved:

**76.90% test accuracy**

---

## 🔮 Single Image Prediction

The `predict.py` script can classify an individual image.

Example output:

```text
----- Prediction -----

Cat probability: 39.74%
Dog probability: 60.26%

Prediction: Dog
Confidence: 60.26%
```

The model uses Softmax to convert its output scores into probabilities.

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/saylimunde/Cat-Dog-Image-Classifier.git
```

### 2. Open the project folder

```bash
cd cat_dog_classifier
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

### 5. Evaluate the model

```bash
python evaluate.py
```

### 6. Check the confusion matrix

```bash
python confusion_matrix.py
```

### 7. Predict a single image

Update the image path inside `predict.py` and run:

```bash
python predict.py
```

---

## 💡 What I Learned

Through this project, I learned how to:

* Build a CNN using PyTorch
* Work with image datasets using `ImageFolder`
* Create training and validation datasets
* Apply image transformations
* Use `DataLoader`
* Implement a training loop
* Calculate training and validation accuracy
* Use `CrossEntropyLoss`
* Use the Adam optimizer
* Understand logits and Softmax probabilities
* Use Dropout to reduce overfitting
* Evaluate a classification model using test data
* Analyze model performance using a confusion matrix
* Build a single-image prediction pipeline
* Manage a machine learning project using Git and GitHub

---

## 🚀 Future Improvements

Possible future improvements include:

* Training for more epochs
* Trying a deeper CNN architecture
* Hyperparameter tuning
* Using a pre-trained model such as ResNet
* Increasing the training dataset
* Adding more carefully selected data augmentation
* Improving prediction confidence calibration

---

## 👩‍💻 Author

**Sayli Munde**

This project was created as part of my journey learning **Machine Learning and Deep Learning with PyTorch**.
