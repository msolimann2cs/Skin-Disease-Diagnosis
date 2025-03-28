#  <p align ="center" height="40px" width="40px"> DermaScreen </p>
##  <p align ="center" height="40px" width="40px"> Skin Lesion Diagnosis using Deep Learning </p>


### <p align ="center"> Implemented using: </p>
<p align ="center">
<a href="https://www.python.org/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/800px-Python-logo-notext.svg.png" width="32" height="32" /></a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer">   <img src="https://opencv.org/wp-content/uploads/2022/05/logo.png" width="32" height="32" /></a>  
<a href="https://keras.io/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/1200px-Keras_logo.svg.png" width="32" height="32" /></a> 
<a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/115px-Tensorflow_logo.svg.png?20170429160244" width="32" height="32" /></a> 
<a href="https://scikit-learn.org/stable/" target="_blank" rel="noreferrer">   <img src="https://e7.pngegg.com/pngimages/309/384/png-clipart-scikit-learn-python-computer-icons-scikit-machine-learning-learning-text-orange.png" width="32" height="32" /></a>  
<a href="https://numpy.org/" target="_blank" rel="noreferrer">   <img src="https://numpy.org/images/logo.svg" width="32" height="32" /></a>  
<a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer">   <img src="https://seaborn.pydata.org/_images/logo-tall-lightbg.svg" width="32" height="32" /></a> 
<a href="https://matplotlib.org/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/2048px-Created_with_Matplotlib-logo.svg.png" width="32" height="32" /></a> 
</p>

<br>
           

<br>

#### <p align = "center"> This project aims to provide a comprehensive skin disease diagnosis and prescreening tool using deep learning techniques. The system is capable of classifying skin diseases into multiple classes with high accuracy, providing probabilities for each class to assist in differential diagnosis. Additionally, it includes support for identifying common skin conditions and integrates the models with a Flutter application to provide a user-friendly interface for both patients and dermatologists. </p>

<br>

##     <p align = "left"> 🎯 Introduction </p>

Skin conditions are a common reason for clinic visits, with an accurate diagnosis being crucial for effective treatment. This project presents a robust deep learning system that analyzes images to identify and diagnose different types of skin lesions.

<br>

##     <p align = "left"> 📚 Dataset </p>
The dataset for the skin cancer portion of the project consists of a total of 41,727 images. These images represent 7 types of skin lesion classes and 1 normal skin class, gathered from public dermatologist datasets and self-collected sources.

<br>

The dataset for the skin condition portion of the project consists of a total of 5,000 images. These images represent 3 types of skin condition classes and 1 normal skin class, gathered from public dermatologist datasets and self-collected sources.

## <p align = "left"> 🗎 Classes  </p>
Skin Disease Classes
AKIEC (Actinic Keratosis)
BCC (Basal Cell Carcinoma)
BKL (Benign Keratosis-like Lesions)
DF (Dermatofibroma)
MEL (Melanoma)
Normal Skin
NV (Melanocytic Nevi)
VASC (Vascular Lesions)
<br>
Supported Skin Conditions
Acne
Eye Bags
Redness
Normal Skin

<br> 

##     <p align = "left"> 🤖 Model </p>
We utilized the mobilenet architecture to create both our models. Trained on the two datasets, we enhanced our training with data augmentation and class balancing for classes with fewer images. Our model achieved an impressive 95% accuracy on the test set. In addition, we assessed our model using various metrics, including the confusion matrix, accuracy and loss histograms, and some with dermatologist classification.

<br>

##     <p align = "left"> 📂 Repository Structure </p>

 -  'skin_cancer': This folder includes the codes for training the skin cancer model and for batch predicting images.

 -  'skin_condition':  This folder includes the codes for training the skin condition model and for batch predicting images.

 -  'augmantation.py': General purpose code for augmanting images to add in our datasets. 

 -  'server':  This folder includes the code the server module that needs to be running for the application.
   
 -  'Clara-Chatbot':  This folder includes the code the chatbot module.

 -  'application':  This folder includes the code the flutter application.


<br>

##     <p align = "left"> 📂 All datasets used links </p>
https://www.kaggle.com/datasets/artakusuma/basedir/data
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
https://www.kaggle.com/datasets/mohamedadel452/ham10000-oversampled-128-128
https://www.kaggle.com/datasets/nayanchaure/acne-dataset/data
https://www.kaggle.com/code/banddaniel/acne-detection-w-kerascv-yolo-v8#-Test--Predictions
https://www.kaggle.com/code/bryamblasrimac/classification-dermatology-vit-accuracy-78-60/notebook
https://www.kaggle.com/datasets/trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes
https://www.kaggle.com/code/osmankagankurnaz/acne-detection-with-yolov8/notebook
https://www.kaggle.com/code/xtvgie/acne-detection/notebook
https://www.kaggle.com/code/stpeteishii/acne-yolo-annotation-view/notebook

<br>

We are still in the process of uploading the our final manually made skin cancer dataset to the repo (41k images)

### <p align ="center"> Do remember to star ⭐ the repository if you like what you see!</p>

---


<div align="center">
  Made with ❤️ by Mohamed Soliman, Abdelrahman Monir, Salma Kaffafy, Laila El Saeed, Karim Noureldin, Ahmed El Geneidy </a>
</div>
