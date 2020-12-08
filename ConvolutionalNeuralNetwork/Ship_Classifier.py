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
from util import func_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import json

#TODO: create validation set
#TODO: Evaluate hyperparameters
#TODO: change activation func
#TODO: Change optimizers
#TODO: Add Noise
print("hi")

with open("C:\\Users\\Sean\\eclipse-workspace\\CS596_FP\\shipsnet.json", 'r') as f:
    data = json.load(f)
    
    
x = np.array(data['data']).astype('uint8')
y = np.array(data['labels']).astype('uint8')




n_spectrum = 3  # color channels (RGB)
weight = 80     #Image weight
height = 80     #Image height
x = x.reshape([-1, weight, height, n_spectrum])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train = np.array(x_train)



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
y_validation = np.array(y_train)


# x = x.reshape([-1, n_spectrum, weight, height])
# pic = x[0]
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
 
# plt.show()

# def add_gaussian_noise(image):  
#     row, col, ch = image.shape
#     mean = 0
#     var = 0.5
#     sigma = var**0.5
#     gauss = np.random.normal(mean,sigma,(row,col,ch))
#     gauss = gauss.reshape(row,col,ch)
#     noisy = image + gauss
#     return noisy
# 
# cnt = 0
# for i in x:
#     x[cnt] = add_gaussian_noise(i)
#     cnt += 1
# 
#  

     
plt.show()

y_validation = to_categorical(y_validation)
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)


#We need to get the labels in the right shape for classification
#y_train = to_categorical(y_train)
#y_test  = to_categorical(y_test)



class CNN_model:
    def __init__(self, input_shape = (80, 80, 3), num_classes = 2, 
                 activation = 'relu', lr = 0.01, pad = 'same', dropout = 0.5, batch = 64):
        self.input_shape = input_shape
        self.num_class = num_classes
        self.act = activation
        self.lr = lr
        self.pad = pad
        self.dropout = dropout
        self.batch = batch
        self.create_model()




    def create_model(self):

        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding= self.pad, input_shape= self.input_shape, activation=self.act))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        model.add(Conv2D(32, (3, 3), padding= self.pad, activation=self.act))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        model.add(Conv2D(32, (10, 10), padding= self.pad, activation= self.act))
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
            x_train, 
            y_train,
            batch_size=32,
            epochs=5,
            validation_split=0.2,
            shuffle=True,
            verbose=2)

        y_pred = model.predict(x_validation)

        y_pred = [np.argmax(pred) for pred in y_pred] 

        print()
        print()
        conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(y_validation, y_pred)
        print("Confusion Matrix: ")
        print(conf_matrix)
        print("Average Accuracy: {}".format(accuracy))
        print("Per-Class Precision: {}".format(precision_array))
        print("Per-Class Recall: {}".format(recall_array))
        


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
    
    
y_pred = model.predict(x_test)

y_pred = [np.argmax(pred) for pred in y_pred] 



print()
print()
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))


#TODO: print an image from data
#TODO: consider image reduction and performance
#TODO: consider color on performance
#TODO: consider num of layers
#TODO: optimize learning rate







