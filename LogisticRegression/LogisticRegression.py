# Data Extraction code, from Akshay Mewada, Kaggle
## IMPORT STATEMENTS FROM KAGGLE
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json  #json file I/O

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Open Data Set

file = open('data/shipsnet.json')
data = json.load(file)
file.close()

Shipsnet= pd.DataFrame(data)
X = np.asarray(data['data']).astype('uint8')
Y = data['labels'] 
#print(Shipsnet.head())
#print(Shipsnet.shape)
#Shipsnet['labels'].value_counts()

# Separate Data into Training and Testing samples

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)

# X_train = preprocessing.scale(X_train)
logreg = LogisticRegression(max_iter=100000)
logreg.fit(X_train, Y_train)


# Evaluation and Performance of ML Model

yHat = sum(logreg.predict(X_test))/len(X_test)
score = logreg.score(X_test,Y_test)
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

func_calConfusionMatrix(logreg.predict(X_test),Y_test)
