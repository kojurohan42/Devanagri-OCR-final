import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
num_classes = 4
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

# Train the model
model.fit(train_generator, epochs=20, validation_data=val_generator)

# Save the trained model
model.save("devanagari_lowerMod_model.h5")
