import cv2
import h5py
import numpy as np
import tensorflow as tf

# Load the model architecture and weights from the HDF5 file
model_path = "devanagari_core_model.h5"  # Replace with the path to your model HDF5 file
model = tf.keras.models.load_model(model_path)

# Load and preprocess the input image
image_path = "chahh.jpg"  # Replace 'input_image.jpg' with the path to your input image
image = cv2.imread(image_path)
# height, width, channels = image.shape

# Print the size
# print(f"Image size: {width}x{height}")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (32, 32))
# image_file = new_image.convert('1')
matrix = np.array(image).astype(int)
reshaped_matrix = matrix.reshape(1, 32, 32, 3)
prediction = np.argmax(model.predict(reshaped_matrix))


# Predict the output
# predictions = model.predict(image)
# predicted_class = np.argmax(predictions)

# Display the predicted class
print("Predicted class:", prediction)
