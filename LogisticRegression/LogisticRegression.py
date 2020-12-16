# Data Extraction code, from Akshay Mewada, Kaggle
# AUTHOR: Jaime Valero Solesio
# CS 596 Machine Learning Project: Binary Classification before/after Noise Introduction
# Fall 2020
## IMPORT STATEMENTS FROM KAGGLE
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json  #json file I/O

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from keras.applications.vgg19 import VGG19
#from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.utils import to_categorical 
from skimage.util import random_noise

# Open Data Set

file = open('/content/drive/MyDrive/shipsnet.json')
data = json.load(file)
file.close()

# Setup of pretrained model (& Adjustment to Image Dimensions)

base_model = VGG19(weights="imagenet", include_top=False,input_shape=(80,80,3))
model = Model(inputs=base_model.input,outputs=base_model.get_layer('block4_pool').output)

Shipsnet= pd.DataFrame(data)
X = np.asarray(data['data']).astype('uint8')
Y = data['labels'] 

X = X / 255.

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

# Split training samples into training and validation (20% Validation)
train_fraction = 0.9
split_point = int(train_fraction * len(X_train))
trainX = X_train[0:split_point]
X_valid = X_train[split_point:]

trainY= Y_train[0:split_point]
Y_valid= Y_train[split_point:]

# Reshape our Arrays
trainX = trainX.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
X_test = X_test.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])

# Logistic Regression Model Training over relevant features
def trainModels():
  global logreg_A,logreg_B,logreg_C,logreg_D
  # Model A
  logreg_A = LogisticRegression(max_iter=10000,fit_intercept=True,solver='sag')
  logreg_A.fit(features_flatten, trainY)
  # Model B
  logreg_B = LogisticRegression(max_iter=10000,fit_intercept=True,solver='newton-cg')
  logreg_B.fit(features_flatten, trainY)
  # Model C
  logreg_C = LogisticRegression(max_iter=10000,fit_intercept=True,solver='liblinear')
  logreg_C.fit(features_flatten, trainY)
  # Model D
  logreg_D = LogisticRegression(max_iter=10000,fit_intercept=True,solver='lbfgs')
  logreg_D.fit(features_flatten, trainY)

def scoreModels():
  # Comparative of Validation Scores before Noise introduction
  ModelA_Score = cross_val_score(logreg_A,X_valid,Y_valid, cv=3)
  ModelB_Score = cross_val_score(logreg_B,X_valid,Y_valid, cv=3)
  ModelC_Score = cross_val_score(logreg_C,X_valid,Y_valid, cv=3)
  ModelD_Score = cross_val_score(logreg_D,X_valid,Y_valid, cv=3)

  print('Model A Score: Hyper Parameters are max_iter=10000,solver=sag'), print("Accuracy: %0.2f (+/- %0.2f)" % (ModelA_Score.mean(), ModelA_Score.std() * 2))
  print('Model B Score: Hyper Parameters are max_iter=10000,solver=newton-cg'), print("Accuracy: %0.2f (+/- %0.2f)" % (ModelB_Score.mean(), ModelB_Score.std() * 2))
  print('Model C Score: Hyper Parameters are max_iter=10000,solver=liblinear'), print("Accuracy: %0.2f (+/- %0.2f)" % (ModelC_Score.mean(), ModelC_Score.std() * 2))
  print('Model D Score: Hyper Parameters are max_iter=10000,solver=lbfgs'), print("Accuracy: %0.2f (+/- %0.2f)" % (ModelD_Score.mean(), ModelD_Score.std() * 2))

# TRAIN PRETRAINED CNN WITHOUT NOISE
# Feature extraction for Model Training
features = model.predict(trainX)
X_test_features = model.predict(X_test)

# Reshape our data after running through CNN VGG19 pretrained Model 
features_flatten = features.reshape((features.shape[0], 5 * 5 * 512))
X_test_flatten = X_test_features.reshape((X_test_features.shape[0], 5 * 5 * 512))

# Train and Evaluate Logistic Regression Models without Noise
print('Training Logistic Regression Models without Gaussian Noise')
trainModels()
print('Model Performance before Noise Introduction on Validation Set')
scoreModels()

# TRAIN PRETRAINED CNN NOISE
# Apply Gaussian Noise to the Training Samples
trainX = random_noise(trainX, mode='gaussian', var=0.05**2)
trainX = (255 * trainX).astype(np.uint8)

# Feature extraction for Model Training
features = model.predict(trainX)
X_test_features = model.predict(X_test)

# Reshape our data after running through CNN VGG19 pretrained Model 

features_flatten = features.reshape((features.shape[0], 5 * 5 * 512))
X_test_flatten = X_test_features.reshape((X_test_features.shape[0], 5 * 5 * 512))

# Train and Evaluate Logistic Regression Models with Noise
print('Training Logistic Regression Models with Gaussian Noise')
trainModels()
print('Model Performance after Noise Introduction on Validation Set')
scoreModels()

# Confusion Matrix & Statistics
def func_calConfusionMatrix(predY, trueY):
    tn, fp, fn, tp = confusion_matrix(trueY,predY).ravel()
    print(confusion_matrix(trueY,predY))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print('Accuracy :', accuracy)
    precisiontn = (tn) / (tn + fp)
    print('Class 0 Precision :', precisiontn)
    precisiontp = (tp) / (tp + fn)
    print('Class 1 Precision :', precisiontp)
    recall = tp / (tp + fn)
    print('Recall :', recall)
    return accuracy,precisiontn,precisiontp, recall

# Evaluation of Most Accurate model using Testing Set
print('Results of Liblinear Logistic Regression Model on Testing Data')
y_pred = logreg_C.predict(X_test_flatten)
func_calConfusionMatrix(y_pred,Y_test)
