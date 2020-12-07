# Data Extraction code, from Akshay Mewada, Kaggle
## IMPORT STATEMENTS FROM KAGGLE
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json  #json file I/O

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.utils import to_categorical 
from skimage.util import random_noise

# Open Data Set

file = open('/content/drive/MyDrive/shipsnet.json')
data = json.load(file)
file.close()


base_model = VGG19(weights="imagenet", include_top=False,input_shape=(80,80,3))
model = Model(inputs=base_model.input,outputs=base_model.get_layer('block4_pool').output)

Shipsnet= pd.DataFrame(data)
X = np.asarray(data['data']).astype('uint8')
Y = data['labels'] 

X = X / 255.

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)

X_train = random_noise(X_train, mode='gaussian', var=0.05**2)
X_train = (255*X_train).astype(np.uint8)
X_train = X_train.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
X_test = X_test.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
# get the features 
features = model.predict(X_train)
features_flatten = features.reshape((features.shape[0], 5 * 5 * 512))

X_test_features = model.predict(X_test)
X_test_flatten = X_test_features.reshape((X_test_features.shape[0], 5 * 5 * 512))
#X_test = preprocessing.scale(X_test)
logreg = LogisticRegression(max_iter=10000,fit_intercept=True)
logreg.fit(features_flatten, Y_train)
# Evaluation and Performance of ML Model
y_pred = logreg.predict(X_test_flatten)
#score = logreg.score(y_pred, Y_test)
#print(score)
yHat = sum(y_pred)/len(X_test)
#score = logreg.score(X_test,Y_test)
print('Results for Logistic Regression')
YtestDiff = np.abs(yHat - Y_test)
avgErr = np.mean(YtestDiff)
stdErr = np.std(YtestDiff)
print('average error: {} ({})'.format(avgErr, stdErr))

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

func_calConfusionMatrix(y_pred,Y_test)

for i in range(len(y_pred)):
  if (y_pred[i] != Y_test[i]):
    image = X_test[i]
    #print(X_test[i])
