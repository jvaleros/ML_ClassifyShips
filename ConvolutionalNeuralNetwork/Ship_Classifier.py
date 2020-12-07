'''
Created on Nov 24, 2020

@author: Sean
'''

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import json

#TODO: create validation set
#TODO: Evaluate hyperparameters
#TODO: change activation func
#TODO: Change optimizers
#TODO: Add Noise


with open("C:\\Users\\Sean\\eclipse-workspace\\CS596_FP\\shipsnet.json", 'r') as f:
    data = json.load(f)
    
    
x = np.array(data['data']).astype('uint8')
y = np.array(data['labels']).astype('uint8')

print(x.shape)


n_spectrum = 3  # color channels (RGB)
weight = 80     #Image weight
height = 80     #Image height
x = x.reshape([-1, weight, height, n_spectrum])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train = np.array(x_train)
print(x_train.shape)

#We need to get the labels in the right shape for classification
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(weight, height, n_spectrum), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# training
model.fit(
    x_train, 
    y_train,
    batch_size=32,
    epochs=18,
    validation_split=0.2,
    shuffle=True,
    verbose=2)


'''To Visualize Data'''
# pic = X[0]
# 
# rad_spectrum = pic[0]
# green_spectrum = pic[1]
# blue_spectum = pic[2]
# 
# plt.figure(2, figsize = (5*3, 5*1))
# plt.set_cmap('jet')
# 
# # show each channel
# plt.subplot(1, 3, 1)
# plt.imshow(rad_spectrum)
# 
# plt.subplot(1, 3, 2)
# plt.imshow(green_spectrum)
# 
# plt.subplot(1, 3, 3)
# plt.imshow(blue_spectum)
#     
# plt.show()

    
    
    
    

#TODO: print an image from data
#TODO: consider image reduction and performance
#TODO: consider color on performance
#TODO: consider num of layers
#TODO: optimize learning rate







