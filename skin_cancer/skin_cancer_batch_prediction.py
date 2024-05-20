import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# Loading the model from path
model_path = '/home/g6/thesis_project/haru_clean_bn_2_93_good.h5'
model = load_model(model_path, custom_objects={'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy})

# Class labels for printing purposes
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'normal_skin', 'nv', 'vasc']

# Function to  preprocess the image
def preprocess_image(image_path, target_size=(380, 380)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict image label
def predict_image(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)[0] * 100  # Convert probabilities to percentages
    
    sorted_probs = predictions[np.argsort(range(len(predictions)))]
    
    total_prob = np.sum(sorted_probs)
    
    # Print predicted classes with their percentages in combination with other classes
    print("Predicted probabilities:")
    for idx, prob in enumerate(sorted_probs):
        class_name = class_labels[idx]
        contribution = prob / total_prob * 100
        print(f"{class_name}: {contribution:.2f}%")
    print()


# Directory containing images
image_dir = "/home/g6/thesis_project/internet_test_images_1"

image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Predicting all images in the directory
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"Image: {image_file}")
    predict_image(image_path, model)