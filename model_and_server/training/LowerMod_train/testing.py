import cv2
import h5py
import numpy as np
import tensorflow as tf

# Load the model architecture and weights from the HDF5 file
model_path = (
    "devanagari_lowerMod_model.h5"  # Replace with the path to your model HDF5 file
)
model = tf.keras.models.load_model(model_path)

# Load and preprocess the input image
image_path = "dash.jpg"  # Replace 'input_image.jpg' with the path to your input image
image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (32, 32))
image = np.expand_dims(image, axis=0)
image = image.astype("float32") / 255.0

# Predict the output
predictions = model.predict(image)
predicted_class = np.argmax(predictions)

# Display the predicted class
print("Predicted class:", predicted_class)
