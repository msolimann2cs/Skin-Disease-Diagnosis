import firebase_admin
from firebase_admin import credentials, db, storage
import os
import numpy as np

from PIL import Image

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy


# CREDENTIALS

cred = credentials.Certificate("/home/g6/thesis_project/gongontest/firebase-sdk.json")
app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://myskin-f6d54-default-rtdb.europe-west1.firebasedatabase.app/',
    'storageBucket': 'myskin-f6d54.appspot.com',
})

bucket = storage.bucket()
# ACNE

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
# Model paths
MODEL_PATH_CONDITION = "/home/g6/thesis_project/acne_bn_3.h5"
MODEL_CONDITION = load_model(MODEL_PATH_CONDITION, custom_objects={'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy})

CLASS_NAMES = ['acne', 'bags','normal' ,'redness']

def preprocess_image_condition(image_path, target_size=(380, 380), mean=0.0, std=1.0):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_image_condition(image_path, model):
    preprocessed_image = preprocess_image_condition(image_path)
    print("acne aaaa")
    predictions = model.predict(preprocessed_image)[0] * 100  
    print("acne peedo")
    sorted_probs = predictions[np.argsort(range(len(predictions)))]
    
    total_prob = np.sum(sorted_probs)
    contributions = " "

    print("Predicted probabilities:")
    for idx, prob in enumerate(sorted_probs):
        class_name = CLASS_NAMES[idx]
        contribution_condition = prob / total_prob * 100
        contributions += str(contribution_condition) + " "

        print(f"{class_name}: {contribution_condition:.2f}%")
    print(contributions)
    return contributions


# Disease

model_path = '/home/g6/thesis_project/haru_clean_bn_2_96_decent_nvbad_mbn.h5'
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'normal_skin', 'nv', 'vasc']
model = load_model(model_path, custom_objects={'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy})

def preprocess_image(image_path, target_size=(380, 380), mean=0.0, std=1.0):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_image(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image) * 100  
    sorted_probs = predictions[np.argsort(range(len(predictions)))]
    total_prob = np.sum(sorted_probs)

    result = ""
    
    print("Predicted probabilities:")
    for idx, prob in enumerate(sorted_probs):
        class_name = class_labels[idx]
        contribution = prob / total_prob * 100

    print()
    return contribution

ref = db.reference('/users')
initial = True

def callback(event):
    global initial
    update = event.data

    print(update)

    if initial:
        print(initial)
        initial = False
    elif str(update.get('result')) == 'READY':
        print("HELLO")

        print('Received update:', update)

        index = update.get('index')
        typeSelected = update.get('type')
        userImage = bucket.get_blob("Patients/patient_" + str(index) + "/picture_0.jpg")
        userImage.download_to_filename("image_" + str(index) + ".jpg")
        if typeSelected == 'condition':
        


            prediction = predict_image("image_" + str(index) + ".jpg", model)
            prediction_condition = predict_image_condition("image_" + str(index) + ".jpg", MODEL_CONDITION)

            fs = ''
            if np.argmax(prediction) != 5:
                print("disease")
                result = ''
                result += str(prediction) + " "
                fs = str(result[1:-2])
            else:
                print("acne")
                result = ''
                result += str(prediction_condition) + " "
                print(prediction_condition)
                fs = str(result[1:-2])

            currentRef = db.reference('/users/'+str(index))
            currentRef.set({
                'username': update.get('username'),
                'email': update.get('email'),
                'index': index,
                'result': fs,
                'type':'condition',
            })
        if typeSelected == 'disease':

            prediction = predict_image("image_" + str(index) + ".jpg", model)
            result = ''
            result += str(prediction) + " "
            print("prediction: " + str(result))

            currentRef = db.reference('/users/'+str(index))
            currentRef.set({
                'username': update.get('username'),
                'email': update.get('email'),
                'index': index,
                'result': str(result[1:-2]),
                'type':'disease',
            })

ref.listen(callback)
