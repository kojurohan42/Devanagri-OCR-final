import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
num_classes = 69
image_size = (32, 32)
batch_size = 32

# Load pre-trained VGG16 model
vgg16 = VGG16(
    weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3)
)

# Freeze VGG16 layers
for layer in vgg16.layers:
    layer.trainable = False

# Create a new model
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

# Print model summary
model.summary()

# Compile the model
model.compile(
    loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"]
)

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=False,
)

# Data augmentation for validation/testing images (only rescaling)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    "Split Dataset/Train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)

# Load validation/testing data
val_generator = val_datagen.flow_from_directory(
    "Split Dataset/Validation",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)

# train the model
history = model.fit(train_generator, epochs=30, validation_data=val_generator)

# Plot training and validation accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "val"], loc="upper left")
plt.show()

# Plot training and validation loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "val"], loc="upper left")
plt.show()

# Generate predictions on validation data
val_generator.reset()
y_val_true = val_generator.classes
y_val_pred = model.predict(val_generator)
y_val_pred = np.argmax(y_val_pred, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_val_true, y_val_pred)


# Plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")


# Get class labels from the generator
class_labels = list(val_generator.class_indices.keys())

# Plot confusion matrix
plot_confusion_matrix(cm, class_labels)

# Save the trained model
model.save("devanagari_core_model.h5")
