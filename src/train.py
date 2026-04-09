"""
Training Script for Fashion MNIST Model
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import os

# Import model
from model import build_model

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Build model
model = build_model()

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Save model
os.makedirs("artifact", exist_ok=True)
model.save("artifact/model.keras")

print("✅ Model trained and saved successfully!")