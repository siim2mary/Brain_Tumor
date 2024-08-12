import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tempfile
import os
from PIL import Image

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


# Load model function
def load_model_from_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        model = tf.keras.models.load_model(tmp_file_path)
        os.remove(tmp_file_path)  # Clean up the temporary file
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Image prediction function
def predict_image(model, image):
    img = image.convert('RGB')
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence_score = np.max(prediction)

    return CLASS_NAMES[predicted_class], confidence_score, prediction


# Load and preprocess data function
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


# Plot confusion matrix function
def plot_confusion_matrix(y_true, y_pred_class):
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)


# Plot ROC curve function
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


# Plot training history function
def plot_training_history(hist):
    # Plot loss
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='Loss')
    plt.plot(hist.history['val_loss'], color='orange', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    st.pyplot(fig)

    # Plot accuracy
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='Accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    st.pyplot(fig)


# Streamlit app
def app():
    st.title("Brain Tumor Classification")

    # Upload the model file
    uploaded_model_file = st.file_uploader("Choose a model file (.h5)", type=["h5"])

    if uploaded_model_file:
        model = load_model_from_uploaded_file(uploaded_model_file)

        if model:
            st.write("Model loaded successfully!")

            # Sidebar options
            st.sidebar.header("Options")
            option = st.sidebar.selectbox("Select an option", [
                "Upload Image",
                "Show Classification Report",
                "Show Confusion Matrix",
                "Show ROC Curve",
                "Show Model Summary",
                "Show Training History",
                "Show Test Accuracy"
            ])

            if option == "Upload Image":
                st.sidebar.header("Upload Image")
                uploaded_image_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

                if uploaded_image_file:
                    st.sidebar.text("Processing...")
                    image = Image.open(uploaded_image_file)
                    st.image(image, caption='Uploaded Image', use_column_width=True)

                    predicted_class_name, confidence_score, prediction = predict_image(model, image)
                    st.sidebar.text(f'Predicted Class: {predicted_class_name}')
                    st.sidebar.text(f'Confidence Score: {confidence_score:.4f}')

            elif option == "Show Classification Report":
                test_data_dir = 'Testing'
                test_data, test_labels = load_and_preprocess_data(test_data_dir)
                y_pred = model.predict(test_data)
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true = test_labels

                st.write("Classification Report:")
                st.text(classification_report(y_true, y_pred_class, target_names=CLASS_NAMES))

            elif option == "Show Confusion Matrix":
                test_data_dir = 'Testing'
                test_data, test_labels = load_and_preprocess_data(test_data_dir)
                y_pred = model.predict(test_data)
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true = test_labels

                st.write("Confusion Matrix:")
                plot_confusion_matrix(y_true, y_pred_class)

            elif option == "Show ROC Curve":
                test_data_dir = 'Testing'
                test_data, test_labels = load_and_preprocess_data(test_data_dir)
                y_pred = model.predict(test_data)
                y_true = test_labels

                roc_auc = plot_roc_curve(y_true, y_pred)
                st.write("ROC AUC Scores:")
                for i, score in roc_auc.items():
                    st.write(f"Class {CLASS_NAMES[i]}: {score:.2f}")

            elif option == "Show Model Summary":
                with st.spinner("Loading model summary..."):
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.write("Model Summary:")
                    st.text("\n".join(model_summary))

            elif option == "Show Training History":
                st.error("Training history is not available. Train the model to view this information.")

            elif option == "Show Test Accuracy":
                test_data_dir = 'Testing'
                test_data, test_labels = load_and_preprocess_data(test_data_dir)
                y_pred = model.predict(test_data)
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true = test_labels

                test_accuracy = np.mean(y_pred_class == y_true)
                st.write(f"Test Accuracy: {test_accuracy:.2f}")


if __name__ == "__main__":
    app()
