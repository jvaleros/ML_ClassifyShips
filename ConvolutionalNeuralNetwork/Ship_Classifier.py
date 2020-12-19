'''
Created on Nov 24, 2020
Description: CNN implementation for Ship image classification project
@author: Sean Tonthat
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import json
from builtins import set


"""
The below path is how I accessed the data.
"""
# with open("C:\\Users\\Sean\\eclipse-workspace\\CS596_FP\\shipsnet.json", 'r') as f:
#     data = json.load(f)

"""
PLEASE READ
USE THE APPROPRIATE PATH IN ORDER TO OBTAIN DATA FROM JSON FILE
"""

with open("shipsnet.json", 'r') as f:
    data = json.load(f)    
    
x = np.array(data['data']).astype('uint8')
y = np.array(data['labels']).astype('uint8')




n_spectrum = 3  # color channels (RGB)
weight = 80     #Image weight
height = 80     #Image height
x = x.reshape([-1, weight, height, n_spectrum])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train = np.array(x_train)



"""
Split up training set into a validation set and a 
smaller training set
"""
S2 = np.random.permutation(len(x_train))
tr = S2[0:2300]

x_train2 = [x_train[i] for i in tr]
x_train2 = np.array(x_train2)


y_train2 = [y_train[i] for i in tr]
y_train2 = np.array(y_train2)


# subsets for validation

x_validation = [x_train[j] for j in S2 if j not in tr]
x_validation = np.array(x_validation)

y_validation = [y_train[j] for j in S2 if j not in tr]
y_validation = np.array(y_validation)





"""
Add Gaussian noise to image
"""
def add_gaussian_noise(image):  
    row, col, ch = image.shape
    mean = 0
    var = 0.5
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy
 
 
cnt = 0
for i in x_train2:
    x_train2[cnt] = add_gaussian_noise(i)
    cnt += 1
  

     

y_train2 = to_categorical(y_train2)
y_test2  = to_categorical(y_test)



"""
The following is a class designed to create an instance of our CNN implementation
Parameters
    (tuple or list) input_shape - shape of an image
    (int) num_classes - the number of labels
    (list) set - can be xvalidation or xtest set
    (list) set2 - can be yvalidation or ytest set
    (str) activation - activation function used
    (float) lr - learning rate
    (str) pad - padding method used
    (float) dropout - dropout method used
    (int) batch - batch size used
    (int) neurons - neurons in hidden layers
    (Boolean) show_images - boolean to show mispredicted images
"""
class CNN_model:
    def __init__(self, input_shape = (80, 80, 3), num_classes = 2, 
                 set = x_validation, set2 = y_validation,
                 activation = 'relu', lr = 0.01, 
                 pad = 'same', dropout = 0.5, 
                 batch = 64, neurons = 32, 
                 show_images = False):
        self.input_shape = input_shape
        self.num_class = num_classes
        self.act = activation
        self.lr = lr
        self.pad = pad
        self.dropout = dropout
        self.batch = batch
        self.neurons = neurons
        self.set = set
        self.set2 = set2
        self.show_images = show_images
        self.create_model()
        self.wrong_cases = [] #cases in which the test or validation set differed from what was predicted



    def create_model(self):

        model = Sequential()

        model.add(Conv2D(self.neurons, (3, 3), padding= self.pad, input_shape= self.input_shape, activation=self.act))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        model.add(Conv2D(self.neurons, (3, 3), padding= self.pad, activation=self.act))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        model.add(Conv2D(self.neurons, (10, 10), padding= self.pad, activation= self.act))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(512, activation= self.act))
        model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr = self.lr, momentum = 0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training
        model.fit(
            x_train2, 
            y_train2,
            batch_size=32,
            epochs=5,
            shuffle=True,
            verbose=2)
        
        y_pred = model.predict(self.set)
        y_pred = [np.argmax(pred) for pred in y_pred] 
        y_pred  = np.array(y_pred)
        self.wrong_cases = np.nonzero(y_pred != y_test)[0]
        
        #Print Confusion Matrix
        print("\nConfusion Matrix")
        target_names = ["no ship", "ship"]
        print(confusion_matrix(self.set2, y_pred, labels=[0,1]))
        print()
        print(classification_report(self.set2, y_pred, target_names=target_names))
        print()
        
        #shows some of the images that were mis-predicted
        if self.show_images == True:
            for i in self.wrong_cases[0:10]:
                pixels = x_test[i]
                plt.title("Label example")
                plt.set_cmap('jet')
                plt.imshow(pixels)
                plt.show()
        



"""
The following loops are meant to test the hyper parameters
"""
 
print()
print("DOING ACTIVATIONS")
activations = ['relu', 'softmax', 'sigmoid']
cnt = 0
for act in activations:
    model = CNN_model( (80,80,3), num_classes = 2, activation = act)
    print("END OF RESULTS FOR activation = ", activations[cnt])
    print()
    cnt += 1
  
print()
print("DOING Learning rate")
lr_arr = [0.01, 0.02, 0.03, 0.04, 0.05]
   
cnt = 0
for lr in lr_arr:
    model = CNN_model( (80,80,3), num_classes = 2, lr = lr)
    print("END OF RESULTS FOR lr = ", lr_arr[cnt])
    cnt += 1
   
print()
print("DOING padding")
pad_arr = ["same", "valid"]
   
   
cnt = 0
for p in pad_arr:
    model = CNN_model( (80,80,3), num_classes = 2, pad = p)
    print("END OF RESULTS FOR padding = ", pad_arr[cnt])
    cnt += 1       
   
print()
print("DOING dropout rate")
do_arr = [0.4, 0.5, 0.6, 0.7, 0.8]
   
cnt = 0
for dr in do_arr:
    model = CNN_model( (80,80,3), num_classes = 2,  dropout = dr)
    print("END OF RESULTS FOR dropout = ", do_arr[cnt])
    cnt += 1   
        
print()
print("DOING batch size")
b_arr = [32, 64, 128, 256]
   
cnt = 0
for b in b_arr:
    model = CNN_model( (80,80,3), num_classes = 2,  batch = b)
    print("END OF RESULTS FOR batch = ", b_arr[cnt])
    cnt += 1      
  
print()
print("DOING Kernel Size") 
n_arr = [32, 64, 128]  
cnt = 0
for b in n_arr:
    model = CNN_model( (80,80,3), num_classes = 2,  neurons = b)
    print("END OF RESULTS FOR batch = ", n_arr[cnt])
    cnt += 1   
     
"""
End of hyperparameter testing
"""


"""
Train final model
"""
model = CNN_model( (80,80,3), num_classes = 2,
                   set = x_test,
                   set2= y_test, 
                   activation = 'relu',
                   lr = 0.02,
                   pad = 'same',
                   dropout = 0.4,
                   batch = 32,
                   neurons = 32,
                   show_images= True)












