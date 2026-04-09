"""
Prediction Script for Fashion MNIST Model
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load model
model = tf.keras.models.load_model("artifact/model.keras")

# Class labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load test data
(_, _), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess
X_test = X_test / 255.0
X_test = X_test.reshape(-1,28,28,1)

# Predict
pred = model.predict(X_test[0].reshape(1,28,28,1))
predicted_class = np.argmax(pred)

print("Predicted:", class_names[predicted_class])
print("Actual:", class_names[y_test[0]])