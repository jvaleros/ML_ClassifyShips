'''
Created on Dec 6, 2020

@author: Sean
'''

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.engine.base_layer import _collect_input_shape
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import json


#This function was taken from: In order to apply gaussian noise to the data set



with open("C:\\Users\\Sean\\eclipse-workspace\\CS596_FP\\shipsnet.json", 'r') as f:
    data = json.load(f)
    
    
x = np.array(data['data']).astype('uint8')
y = np.array(data['labels']).astype('uint8')



n_spectrum = 3  # color channels (RGB)
weight = 80     #Image weight
height = 80     #Image height

x = x.reshape([-1, weight, height, n_spectrum]) #Input shape should not be (80,80, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#code borrowed from 
#https://stackoverflow.com/questions/22937589/
#how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

#COPY THIS PART
def add_gaussian_noise(image):  
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

cnt = 0
for i in x_train:
    x_train[cnt] = add_gaussian_noise(i)
    cnt += 1
    




x_train = np.array(x_train)
y_train = np.array(y_train)
x_test  = np.array(x_test)
y_test  = np.array(y_test)



# y_validation = to_categorical(y_validation)

#We need to get the labels in the right shape for classification



y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)


class CNN_model:
    def __init__(self, input_shape = (80, 80, 3), num_classes = 2, 
                 activation = 'relu', lr = 0.01, pad = 'same', dropout = 0.5):
        self.input_shape = input_shape
        self.num_class = num_classes
        self.act = activation
        self.lr = lr
        self.pad = pad
        self.dropout = dropout
        self.create_model()
      
          
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=(80,80,3),filters=64,kernel_size=(3,3), padding= self.pad, activation=self.act))
        model.add(Conv2D(filters=64,kernel_size=(3,3), padding=self.pad, activation=self.act))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding= self.pad, activation= self.act))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2))) 
            
        model.add(Flatten())
        model.add(Dense(4096, activation= self.act))
        model.add(Dropout(self.dropout))
        model.add(Dense(4096, activation= self.act))
        model.add(Dropout(self.dropout))
        model.add(Dense(2, activation= 'softmax'))
        sgd = SGD(lr = self.lr, momentum = 0.9, nesterov = True)
        model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
              
        model.fit(x_train, y_train, batch_size=32, epochs=3, validation_split=0.2, shuffle=True, verbose=1)
        #self.model = model
        
        
    def score2(self, x_validation, y_validation):
        return self.model.score(x_validation, y_validation)



print()
print("DOING ACTIVATIONS")
activations = ['relu', 'softmax', 'sigmoid']

cnt = 0
for act in activations:
    model = CNN_model( (80,80,3), num_classes = 2, activation = act)
    print("END OF RESULTS FOR activation = ", activations[cnt])
    cnt += 1

print()
print("DOING Learning rate")
lr_arr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

cnt = 0
for lr in lr_arr:
    model = CNN_model( (80,80,3), num_classes = 2, lr = lr)
    print("END OF RESULTS FOR lr = ", lr_arr[cnt])
    cnt += 1

print()
print("DOING padding")
pad_arr = ["same", "valid"]

cnt = 0
for p in activations:
    model = CNN_model( (80,80,3), num_classes = 2, pad = p)
    print("END OF RESULTS FOR padding = ", pad_arr[cnt])
    cnt += 1       

print()
print("DOING dropout rate")
do_arr = ["same", "valid"]

cnt = 0
for dr in do_arr:
    model = CNN_model( (80,80,3), num_classes = 2,  dropout = dr)
    print("END OF RESULTS FOR dropout = ", pad_arr[cnt])
    cnt += 1   
     
     
     
    
    
    
    








