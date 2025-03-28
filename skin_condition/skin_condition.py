from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import InceptionV3, MobileNet
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, BackupAndRestore, TensorBoard
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D
from keras.applications.inception_v3 import preprocess_input  
from keras import regularizers
from keras.regularizers import l2
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

#directories for the datasets from disk
train_path='/home/g6/thesis_project/beedzACNE/acne-data/final_temp_with_test/train'
valid_path='/home/g6/thesis_project/beedzACNE/acne-data/final_temp_with_test/test'

#hyper parameters
train_batch_size = 64
val_batch_size = 10
image_size = 380
epochs = 20
#normalization
rescale_factor = 1.0 / 255.0

datagen = ImageDataGenerator(rescale=rescale_factor)
#loading data from disk using flow from directory to help with memory issues
train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size, image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size, image_size),
                                            batch_size=val_batch_size, shuffle=False)

# creating base mobilenet model
base_model=MobileNet(include_top=False,input_shape=(image_size,image_size,3))

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu',kernel_regularizer=l2(0.01))(x)
x = Dropout(0.25)(x)
predictions = Dense(4, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)


early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=4, verbose=1, min_delta=0.01, mode='max')
# learning rate reduction on val accuracy with hyper paramenters
reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [reduce_lr]

# functions to return top2/3 accuracy which is relevant in our case because we want to ensure our correct label is at most one of the top 3 labels given by the model
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

# Custom class weights to try and force the model to give better results towards certain classes
class_weights={
    0:1.0, #acne
    1:2.5, #bags # try to make the model more sensitive to bags
    2:1.0, #normal
    3:1.0 #redness
}
# train the model
history = model.fit(train_batches,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    validation_data=valid_batches,
                    class_weight=class_weights,
                    epochs=epochs, verbose=1,
                    callbacks=callbacks_list)
# saving the model
model_path = '/home/g6/thesis_project/acne_bn_3.h5'
model.save(model_path)
print("Model saved successfully at:", model_path)

# plotting the results and evaluating the model using confusion matrix, f1 score, and AuC score

plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
y_true = valid_batches.classes
y_pred = model.predict(valid_batches)
y_pred_classes = np.argmax(y_pred, axis=1)

# F1 Score
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print("F1 Score: ", f1)

# AUC Score
y_pred_proba = y_pred
auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
print("AUC Score: ", auc)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot F1 Score and AUC Score
scores = {'F1 Score': f1, 'AUC Score': auc}
plt.figure(figsize=(8, 6))
plt.bar(scores.keys(), scores.values(), color=['blue', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('F1 Score and AUC Score')
plt.ylim(0, 1)
plt.show()