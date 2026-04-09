"""
CNN Model Architecture for Fashion MNIST
"""
"""
CNN Model Architecture for Fashion MNIST
"""

import tensorflow as tf

def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28,28,1)),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model