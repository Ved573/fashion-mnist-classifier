import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("artifact/model.keras")

# Class labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.write("""
This app uses a CNN model trained on the Fashion MNIST dataset.

- Model: CNN
- Accuracy: ~90%
- Framework: TensorFlow/Keras
""")

# Title
st.title("👕 Fashion MNIST Classifier (Deep Learning App)")
st.write("Upload an image and get predictions with confidence scores")

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28,1)

    st.divider()

    # Prediction
    prediction = model.predict(img_array)[0]

    # Top 3 predictions
    top3 = np.argsort(prediction)[-3:][::-1]

    st.subheader("🔮 Top Predictions")

    for i in top3:
        st.progress(float(prediction[i]))
        st.write(f"👉 {class_names[i]} : {prediction[i]*100:.2f}%")