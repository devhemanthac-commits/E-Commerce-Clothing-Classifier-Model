# E-Commerce Visual Classification (Fashion MNIST CNN)

This repository contains a core computer vision feature designed for an e-commerce platform. It leverages a Convolutional Neural Network (CNN) built with TensorFlow and Keras to automatically classify raw images of clothing items into 10 distinct categories.

This feature is designed to improve visual search, automated inventory management, and recommendation engines.

## Problem Statement
The platform needs an automated way to classify clothing items from 28x28 grayscale images. The dataset used is the industry-standard **Fashion MNIST** dataset, which contains 70,000 images spread across 10 classes (e.g., T-shirts, Trousers, Sneakers, Bags).

## Convolutional Neural Network (CNN) Architecture

Unlike standard machine learning models, this CNN automatically extracts complex visual patterns (like edges, textures, and shapes) using specialized layers.

**Phase 1: Data Pipeline & Preprocessing**
- Automates the downloading and loading of the `fashion_mnist` dataset via Keras.
- Normalizes pixel values from a `[0, 255]` scale down to `[0, 1]` for optimal Neural Network gradient descent.
- Reshapes the arrays to explicitly include the single color channel `(28, 28, 1)` required by 2D Convolutional layers.

**Phase 2: Network Design (Sequential API)**
1. **First Convolutional Block:** A `Conv2D` layer (32 filters, 3x3 kernel, ReLU activation) followed by a `MaxPooling2D` layer to extract low-level features (e.g., edges, basic shapes).
2. **Second Convolutional Block:** A `Conv2D` layer (64 filters, 3x3 kernel, ReLU activation) followed by another `MaxPooling2D` layer to extract complex, high-level features (e.g., textures, item parts).
3. **Dense Classifier:** A `Flatten` layer converts the 2D feature maps into a 1D vector, passed into a fully-connected `Dense` layer (128 neurons).
4. **Regularization:** A `Dropout` layer (rate: `0.2`) randomly deactivates neurons during training to heavily mitigate overfitting on the training data.
5. **Output Layer:** A final `Dense` layer (10 neurons) using the `softmax` activation function to output a clean probability distribution across the 10 clothing categories.

**Phase 3: Compilation & Smart Training**
- **Optimizer:** Adaptive Moment Estimation (`adam`).
- **Loss Function:** `sparse_categorical_crossentropy` (ideal for integer-encoded labels).
- **EarlyStopping Callback:** Monitors validation loss (`val_loss`) during training and automatically stops the process if the model stops generalizing to new data (patience = 3 epochs). Furthermore, it automatically restores the best optimal weights.

## How to Run the Pipeline

1. **Install Requirements:**
   Ensure you have the necessary deep learning libraries installed in your Python environment:
   ```bash
   pip install tensorflow numpy
   ```

2. **Execute the Script:**
   ```bash
   python fashion_mnist_cnn.py
   ```

## Evaluation & Business Output
Running the script automates the entire process: downloading the data, structuring the CNN, and initiating the smart training sequence. 

Once training halts, the system evaluates the model against an unseen test set to generate a final **Business Accuracy Metric**. Finally, it selects a random image from the test set and performs a live prediction, printing the *Predicted Category Name* side-by-side with the *Actual Category Name* to demonstrate real-time functionality.