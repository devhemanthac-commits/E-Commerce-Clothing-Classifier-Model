import tensorflow as tf
from tensorflow import keras
import numpy as np

# Phase 1: Data Pipeline & Preprocessing
def load_and_preprocess_data():
    """
    Loads the Fashion MNIST dataset and preprocesses it for a CNN.
    """
    print("--- Phase 1: Data Pipeline & Preprocessing ---")
    
    # 1. Load the Fashion MNIST dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Define the class names for later visualization/evaluation
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f"Loaded {len(train_images)} training images and {len(test_images)} testing images.")

    # 2. Normalize the pixel values (scale them between 0 and 1)
    # Neural networks prefer small inputs. Since pixel values range from 0 to 255, 
    # dividing by 255.0 scales them to the [0, 1] range, improving gradient descent convergence.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 3. Reshape arrays to include the color channel
    # Convolutional layers expect images with explicit color channels (batch_size, height, width, channels).
    # Since Fashion MNIST is grayscale, it has 1 color channel. We explicitly reshape to add this dimension: (28, 28, 1).
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    print("Data normalized and reshaped to include the color channel (28, 28, 1).\n")
    return (train_images, train_labels), (test_images, test_labels), class_names


# Phase 2: CNN Architecture Design
def build_cnn_model():
    """
    Constructs the CNN architecture using Keras Sequential API.
    """
    print("--- Phase 2: CNN Architecture Design ---")
    
    # Build a Sequential model
    model = keras.Sequential([
        # First Convolutional Block
        # Conv2D with 32 filters (3x3 kernel). Scans the image for low-level features like edges.
        # Activation 'relu' (Rectified Linear Unit) introduces non-linearity to learn complex patterns without vanishing gradients.
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # MaxPooling (2x2) downsamples feature maps, reducing computation and adding spatial invariance.
        keras.layers.MaxPooling2D(2, 2),

        # Second Convolutional Block
        # 64 filters capture higher-level, more complex features (like specific textures or object parts).
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        # Flatten layer transitions from 2D convolutional feature maps to a 1D vector required by Dense layers.
        keras.layers.Flatten(),

        # Dense Hidden Layer
        # Fully connected layer with 128 neurons to combine and interpret extracted visual features.
        keras.layers.Dense(128, activation='relu'),
        
        # Dropout layer (rate = 0.2 means randomly turning off 20% of neurons during each training step).
        # This forces the network to learn redundant representations, heavily reducing overfitting to the training data.
        keras.layers.Dropout(0.2),

        # Final Output Layer
        # 10 neurons corresponding to the 10 fashion categories. 
        # Softmax activation converts unscaled logits into a clean probability distribution (summing to 1.0).
        keras.layers.Dense(10, activation='softmax')
    ])
    
    print("Model Architecture Summary:")
    model.summary()
    return model


# Phase 3: Compilation & Smart Training
def compile_and_train(model, train_images, train_labels):
    """
    Compiles the model and trains it using an EarlyStopping callback with validation tracking.
    """
    print("\n--- Phase 3: Compilation & Smart Training ---")
    
    # Compile the model
    # 'adam' adaptively tunes learning rates, converging faster than standard SGD.
    # 'sparse_categorical_crossentropy' is used because labels are integers (0-9) rather than one-hot encoded arrays.
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Implement Keras EarlyStopping
    # Instead of arbitrarily guessing how many epochs to train for, we monitor validation loss. 
    # If val_loss stops improving for 3 consecutive epochs ('patience=3'), training stops.
    # 'restore_best_weights=True' ensures we revert to the optimal model weights, preventing overfitting.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True,
        verbose=1
    )

    print("Starting training with EarlyStopping (monitoring val_loss)...")
    
    # Train the model, keeping 20% of the training data aside as a validation split to track generalization.
    history = model.fit(
        train_images, 
        train_labels, 
        epochs=30, # Safe to set high because EarlyStopping will halt it appropriately
        validation_split=0.2, 
        callbacks=[early_stopping]
    )
    return history


# Phase 4: Evaluation & Business Output
def evaluate_and_predict(model, test_images, test_labels, class_names):
    """
    Evaluates the model on test data and demonstrates a single real-time prediction side-by-side.
    """
    print("\n--- Phase 4: Evaluation & Business Output ---")
    
    # 1. Evaluate overall accuracy on the unseen test set
    print("Evaluating final model on test data...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\n[Business Metric] Final Test Accuracy: {test_acc * 100:.2f}%\n")

    # 2. Real-time prediction demonstration
    import random
    random_index = random.randint(0, len(test_images) - 1)
    
    # Run prediction on a single random image (requires slicing to keep batch dimension)
    prediction_probs = model.predict(test_images[random_index:random_index+1])
    
    # Extract predicted label and actual label
    predicted_class_index = np.argmax(prediction_probs[0])
    predicted_label = class_names[predicted_class_index]
    actual_label = class_names[test_labels[random_index]]

    print("=" * 60)
    print(" REAL-TIME E-COMMERCE VISUAL CLASSIFICATION DEMONSTRATION")
    print("=" * 60)
    print(f"| Predicted Category Name: \t{predicted_label:15} |")
    print(f"| Actual Category Name:    \t{actual_label:15} |")
    print("=" * 60)
    
    if predicted_label == actual_label:
        print("-> [SUCCESS] Model correctly identified the item out of 10 categories!")
    else:
        print("-> [FAILURE] Model misclassified the item.")


def main():
    print("*" * 60)
    print(" E-COMMERCE FASHION IMAGE CLASSIFIER (CNN) PIPELINE")
    print("*" * 60)
    
    # Phase 1: Load and Preprocess
    (train_images, train_labels), (test_images, test_labels), class_names = load_and_preprocess_data()
    
    # Phase 2: Design Architecture
    model = build_cnn_model()
    
    # Phase 3: Compile and Train
    _ = compile_and_train(model, train_images, train_labels)
    
    # Phase 4: Evaluate Predict
    evaluate_and_predict(model, test_images, test_labels, class_names)
    
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
