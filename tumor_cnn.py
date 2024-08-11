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
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

# %%
#!pip install tensorflow tensorflow-gpu opencv-python matplotlib

# %%
gpus =tf.config.experimental.list_physical_devices('CPU')

# %%
#Avoid OOM errors by setting gpu consumption growth
gpus =tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True)

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

#Access the Training Directory
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
        folder_path = os.path.join(extract_dir,folder)
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

# %% [markdown]
# DATA PREPROCESSING
# **************************************

# %%
# Normalize the data
train_data = train_data / 255.0  # Normalize train data to range [0, 1]
test_data = test_data / 255.0  # Normalize test data to range [0, 1]
# %% [markdown]
# SPLIT THE DATA
# *******************************

# %%
# Split the train data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels)  # Split train data into training and validation sets

# Split the test data into test set
x_test = test_data  # Test data remains unchanged
y_test = test_labels  # Test labels remain unchanged

# %% [markdown]
# LABEL CONVERSION
# **************************

# %%
# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)  # Convert train labels to categorical
y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)  # Convert validation labels to categorical
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)  # Convert test labels to categorical

# %% [markdown]
# MODEL DEFINITION
# *********************

# %%
# Define the model
model = Sequential([  # Define a sequential model
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),  # Convolutional layer with 32 filters
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer with 64 filters
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    Conv2D(128, (3, 3), activation='relu'),  # Convolutional layer with 128 filters
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    Flatten(),  # Flatten layer
    Dense(256, activation='relu'),  # Dense layer with 256 units
    Dropout(0.5),  # Dropout layer with 50% dropout rate
    Dense(4, activation='softmax')  # Output layer with 4 units and softmax activation
])

# %% [markdown]
# MODEL COMPILATION
# ***********************

# %%
# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model with Adam optimizer and categorical crossentropy loss

# %% [markdown]
# MODEL TRAINING
# ************************************

# %%
# Train the model
hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))  # Train the model on training data with validation on validation data

# %%
hist.history

# %% [markdown]
# PLOT PERFORMANCE OF TRAIN DATA
# ************************************************************************************

# %%
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.title('Validation loss of Train data')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# %%
#visualise accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
plt.title('Accuracy of Train')
fig.suptitle('Accuracy', fontsize=20)

# %% [markdown]
# MODEL EVALUATION
# **************************************************************

# %%
# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(x_val, y_val)  # Evaluate the model on validation data
print(f'Validation accuracy: {val_acc:.2f}')  # Print validation accuracy

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)  # Evaluate the model on test data
print(f'Test accuracy: {test_acc:.2f}')  # Print test accuracy

# %% [markdown]
# MODEL PREDICTION
# *********************************

# %%
# Make predictions
y_pred = model.predict(x_test)  # Make predictions on test data

# Convert predictions to class labels
y_pred_class = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
y_true = np.argmax(y_test, axis=1)  # Get true class labels

# %% [markdown]
# MODEL METRICS
# *********************************************

# %%
# Print classification report
print(classification_report(y_true, y_pred_class))  # Print classification report

# Print confusion matrix
print(confusion_matrix(y_true, y_pred_class))  # Print confusion matrix

# %% [markdown]
# CONFUSION MATRIX PLOT
# *******************************************************************

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Print confusion matrix
cm = confusion_matrix(y_true, y_pred_class)

# Plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# ROC-AUC SCORE
# **************************

# %%
# Calculate ROC-AUC score
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_auc:.2f}')

# %% [markdown]
# SAVE THE MODEL
# ***********************

# %%
# Save the model
model.save('CNN_image_classification_model.h5')

# %% [markdown]
# DOING TEST ON AN UNSEEN IMAGE
# *********************************

# %%
# Load unseen image
from PIL import Image
import numpy as np

img_path = 'Testing/meningioma/Te-me_0012.jpg'
img = Image.open(img_path)
img = img.resize((img_height, img_width))
img_array = np.array(img)
img_array = img_array / 255.0

# %% [markdown]
# DISPLAY UNSEEN IMAGE
# **************************

# %%
# Display unseen image
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title('Unseen Image')
plt.show()

# %% [markdown]
# PREDICT CLASS CORRECTLY
# ************************

# %%
# Import necessary libraries
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('CNN_image_classification_model.h5')

# Load image
img = Image.open('Testing/meningioma/Te-me_0012.jpg')

# Resize image
img = img.resize((224, 224))

# Convert image to array
img_array = np.array(img)

# Reshape array
img_array = np.expand_dims(img_array, axis=-1)
img_array = np.repeat(img_array, 3, axis=-1)
img_array = img_array.reshape((1, 224, 224, 3))

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Get class name
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
predicted_class_name = class_names[predicted_class]

# Print predicted class
print(f'Predicted Class: {predicted_class_name}')

# %%
print(f'Predicted Class: {class_names[predicted_class]}')

# %%
from PIL import Image
import matplotlib.pyplot as plt

# Load new image
new_img = Image.open('Testing/glioma/Te-gl_0013.jpg')


# Display loaded image
plt.figure(figsize=(6, 6))
plt.imshow(new_img)
plt.title('Loaded Image')
plt.show()

# Resize new image
new_img = new_img.resize((224, 224))  # Assuming original image size is (224, 224)

# Convert new image to RGB (if necessary)
new_img = new_img.convert('RGB')

# Preprocess new image (if necessary)
new_img_array = np.array(new_img) / 255.0  # Normalize pixel values

# Make prediction with new image')

# Display loaded image
plt.figure(figsize=(6, 6))
plt.imshow(new_img)
plt.title('Loaded Image')
plt.show()

# Resize new image
new_img = new_img.resize((224, 224))  # Assuming original image size is (224, 224)

# Convert new image to RGB (if necessary)
new_img = new_img.convert('RGB')

# Preprocess new image (if necessary)
new_img_array = np.array(new_img) / 255.0  # Normalize pixel values

# Make prediction with new image
prediction = model.predict(new_img_array.reshape((1, 224, 224, 3)))

# Get the predicted class index
predicted_class_index = np.argmax(prediction)

# Map the predicted class index to the class name
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
predicted_class_name = class_names[predicted_class_index]

print(f'Predicted Class: {predicted_class_name}')
