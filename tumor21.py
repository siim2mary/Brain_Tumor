import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import os

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load the trained model
model = load_model('CNN_image_classification_model.h5')


def predict_image(image):
    img = image.convert('RGB')
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction)

    return CLASS_NAMES[predicted_class], confidence_score, prediction


def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    label_dict = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for label_name in CLASS_NAMES:
        label_dir = os.path.join(data_dir, label_name)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label_dict[label_name])

    return np.array(images), np.array(labels)


def plot_confusion_matrix(y_true, y_pred_class):
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)


def plot_roc_curve(y_true, y_score):
    plt.figure(figsize=(10, 8))
    roc_auc = {}
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = roc_auc_score(y_true == i, y_score[:, i])
        plt.plot(fpr, tpr, label=f'Class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    return roc_auc


def app():
    st.title("Brain Tumor Classification")

    if 'results' not in st.session_state:
        st.session_state.results = {}

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.sidebar.text("Processing...")
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        predicted_class_name, confidence_score, prediction = predict_image(image)
        st.sidebar.text(f'Predicted Class: {predicted_class_name}')
        st.sidebar.text(f'Confidence Score: {confidence_score:.4f}')

        # Store results
        st.session_state.results['prediction'] = (predicted_class_name, confidence_score)

    if st.sidebar.button('Show Classification Report'):
        # Load and preprocess test data
        test_data_dir = 'Testing'
        test_data, test_labels = load_and_preprocess_data(test_data_dir)
        y_pred = model.predict(test_data)
        y_pred_class = np.argmax(y_pred, axis=1)

        if len(test_labels.shape) > 1 and test_labels.shape[1] == NUM_CLASSES:
            y_true = np.argmax(test_labels, axis=1)
        else:
            y_true = test_labels

        # Compute and display classification report
        st.session_state.results['classification_report'] = classification_report(y_true, y_pred_class,
                                                                                  target_names=CLASS_NAMES)
        st.write("Classification Report:")
        st.text(st.session_state.results['classification_report'])

    if st.sidebar.button('Show Confusion Matrix'):
        # Load and preprocess test data
        test_data_dir = 'Testing'
        test_data, test_labels = load_and_preprocess_data(test_data_dir)
        y_pred = model.predict(test_data)
        y_pred_class = np.argmax(y_pred, axis=1)

        if len(test_labels.shape) > 1 and test_labels.shape[1] == NUM_CLASSES:
            y_true = np.argmax(test_labels, axis=1)
        else:
            y_true = test_labels

        # Compute and display confusion matrix
        st.session_state.results['confusion_matrix'] = (y_true, y_pred_class)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y_true, y_pred_class)

    if st.sidebar.button('Show Test Accuracy'):
        # Load and preprocess test data
        test_data_dir = 'Testing'
        test_data, test_labels = load_and_preprocess_data(test_data_dir)
        y_pred = model.predict(test_data)
        y_pred_class = np.argmax(y_pred, axis=1)

        if len(test_labels.shape) > 1 and test_labels.shape[1] == NUM_CLASSES:
            y_true = np.argmax(test_labels, axis=1)
        else:
            y_true = test_labels

        # Compute and display test accuracy
        test_accuracy = np.mean(y_pred_class == y_true)
        st.session_state.results['test_accuracy'] = test_accuracy
        st.write(f"Test Accuracy: {test_accuracy:.4f}")

    if st.sidebar.button('Show ROC Curve'):
        # Load and preprocess test data
        test_data_dir = 'Testing'
        test_data, test_labels = load_and_preprocess_data(test_data_dir)
        y_pred = model.predict(test_data)

        if len(test_labels.shape) > 1 and test_labels.shape[1] == NUM_CLASSES:
            y_true = np.argmax(test_labels, axis=1)
        else:
            y_true = test_labels

        # Compute and display ROC curve
        roc_auc = plot_roc_curve(y_true, y_pred)
        st.session_state.results['roc_auc'] = roc_auc

        # Display ROC AUC scores
        st.write("ROC AUC Scores:")
        for i, score in roc_auc.items():
            st.write(f"Class {CLASS_NAMES[i]}: {score:.2f}")


if __name__ == "__main__":
    app()
