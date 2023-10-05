
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import seaborn as sns
import numpy as np

# Assuming your log file has columns like 'epoch', 'loss', and 'accuracy'
# You might need to modify this code depending on your log format
with open('Models/Core_AlexNet_model/training.log', 'r') as f:
    lines = f.readlines()

epochs = []
losses = []
accuracies = []
val_losses = []
val_accuracies = []

for line in lines:
    parts = line.strip().split(',')
    epoch = int(parts[0])
    loss = float(parts[2])
    accuracy = float(parts[1])
    val_loss = float(parts[5])
    val_accuracy = float(parts[4])
    
    epochs.append(epoch)
    losses.append(loss)
    accuracies.append(accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
 
# Create subplots for loss and accuracy
# Generate accuracy and loss graphs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()



# trainDataGen = ImageDataGenerator(rotation_range=5,
#                                   width_shift_range=0.1,
#                                   height_shift_range=0.1,
#                                   rescale=1.0 / 255,
#                                   shear_range=0.2,
#                                   zoom_range=0.2,
#                                   horizontal_flip=False,
#                                   fill_mode='nearest')

testDataGen = ImageDataGenerator(rescale=1.0 / 255)

# trainGenerator = trainDataGen.flow_from_directory(os.path.join('Split Dataset', 'Train'),
#                                                   target_size=(32, 32),
#                                                   batch_size=32,
#                                                   color_mode='grayscale',
#                                                   classes=[str(Class)
#                                                            for Class in range(69)],
#                                                   class_mode='categorical')

validationGenerator = testDataGen.flow_from_directory(os.path.join('Core_Split DataSet', 'Validation'),
                                                      target_size=(32, 32),
                                                      batch_size=32,
                                                      color_mode='grayscale',
                                                      classes=[
                                                          str(Class) for Class in range(69)],
                                                      class_mode='categorical')

# Load the best model
model = load_model(os.path.join('Models/Core_AlexNet_model', 'best_val_loss.hdf5'))

# Evaluate the model
loss, acc = model.evaluate(validationGenerator)
print('Loss on Validation Data:', loss)
print('Accuracy on Validation Data:', '{:.4%}'.format(acc))

# Confusion matrix
class_names = [str(Class) for Class in range(69)]

validationGenerator.reset() 
y_true = validationGenerator.classes
Y_pred = model.predict(validationGenerator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()