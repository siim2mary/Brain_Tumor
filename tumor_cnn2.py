# %%
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# %%
#!pip install tensorflow tensorflow-gpu opencv-python matplotlib

# %%
gpus = tf.config.experimental.list_physical_devices('CPU')

# %%
# Avoid OOM errors by setting GPU consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# %%
#!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

import zipfile
import os

# Define the path to the ZIP file
zip_file_path = r'C:\Users\Joby\PycharmProjects\pythonProject1Brain_Tumor\brain-tumor-mri-dataset.zip'

# Define the extraction directory (same as base directory or a new one)
extract_dir = r'C:\Users\Joby\PycharmProjects\pythonProject1Brain_Tumor'

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction completed.")

# Access the Training Directory
# Define the base directory where the dataset was extracted
base_dir = r'C:\Users\Joby\PycharmProjects\pythonProject1Brain_Tumor'

# Path to the 'Training' directory
train_data_dir = os.path.join(base_dir, 'Training')
# Path to the 'Testing' directory
test_data_dir = os.path.join(base_dir, 'Testing')

train_data_dir = 'Training'
test_data_dir = 'Testing'

# %%
# Define image size
img_height, img_width = 224, 224

# %%
# Load and preprocess images
def load_images_and_labels(extract_dir):
    labels_dict = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
    data = []
    labels = []

    for folder in os.listdir(extract_dir):
        folder_path = os.path.join(extract_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = load_img(img_path, target_size=(img_height, img_width))
                img = img_to_array(img)
                data.append(img)
                labels.append(labels_dict[folder])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(data), np.array(labels)

# %%
# Load train data and labels
train_data, train_labels = load_images_and_labels(train_data_dir)

# %%
# Load test data and labels
test_data, test_labels = load_images_and_labels(test_data_dir)

# %%
# Normalize the data
train_data = train_data / 255.0  # Normalize train data to range [0, 1]
test_data = test_data / 255.0  # Normalize test data to range [0, 1]

# %%
# Split the train data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)
y_test = tf.keras.utils.to_categorical(test_labels, num_classes=4)

# %%
# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# %%
# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# Train the model
hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# %%
# Plot loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.title('Validation loss of Train data')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# %%
# Plot accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
plt.title('Accuracy of Train')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# %%
# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(x_val, y_val)
print(f'Validation accuracy: {val_acc:.2f}')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_data, y_test)
print(f'Test accuracy: {test_acc:.2f}')


# %%
# Make predictions
y_pred = model.predict(test_data)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# %%
# Print classification report
print(classification_report(y_true, y_pred_class))

# Print confusion matrix
print(confusion_matrix(y_true, y_pred_class))

# %%
# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_class)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# %%
# Calculate ROC-AUC score
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_auc:.2f}')

# %%
# Save the model
model.save('CNN_image_classification_model.h5')

# Compress the model file into a .tar.gz archive
import tarfile

model_file = 'CNN_image_classification_model.h5'
tar_file = 'CNN_image_classification_model.tar.gz'

with tarfile.open(tar_file, 'w:gz') as tar:
    tar.add(model_file, arcname=os.path.basename(model_file))

print(f'Model compressed to: {tar_file}')

# %%
# Load unseen image
from PIL import Image

img_path = 'Testing/meningioma/Te-me_0012.jpg'
img = Image.open(img_path)
img = img.resize((img_height, img_width))
img_array = np.array(img) / 255.0

# %%
# Display unseen image
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title('Unseen Image')
plt.show()

# %%
# Predict class for unseen image
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Map the predicted class index to the class name
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
predicted_class_name = class_names[predicted_class]
print(f'Predicted Class: {predicted_class_name}')

# %%
# Load new image for testing
new_img = Image.open('Testing/glioma/Te-gl_0013.jpg')
new_img = new_img.resize((img_height, img_width))
new_img_array = np.array(new_img) / 255.0

# Display loaded image
plt.figure(figsize=(6, 6))
plt.imshow(new_img)
plt.title('Loaded Image')
plt.show()

# %%
# Predict class for new image
new_img_array = new_img_array.reshape((1, img_height, img_width, 3))
prediction = model.predict(new_img_array)
predicted_class_index = np.argmax(prediction)
predicted_class_name = class_names[predicted_class_index]
print(f'Predicted Class: {predicted_class_name}')
