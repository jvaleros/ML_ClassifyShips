# Noise Introduction to Binary Classification Models on Kaggle Ship Image Dataset 
## Description
 This project is part of a Machine Learning course at San Diego State University. In it, we chose to compare the robustness of three different supervised learning models(SVM, CNN, Log Reg) on the Kaggle ship-image dataset for binary classification (ship or no ship) after introducing noise to the dataset.
 
## Dataset
![Samples Images from Dataset](https://github.com/jvaleros/ML_ClassifyShips/blob/main/images/ShipsSampleImage.png)
 The dataset we are using is located at [Kaggle Ships in Satellite Imagery](https://www.kaggle.com/rhammell/ships-in-satellite-imagery). 
 This dataset contains 4000 png files, with the same resolution. Moreover, to simplify processing, the dataset contains a JSON file with data and the ground truth labels for each image in the set.
## Machine Learning Models
### CNN
Highest accuracy model on test data had hyperparameters: Activation = ‘relu’, Lr = 0.02, 
Padding = ‘same’, dropout = 0.5, batch size =64. Accuracy was = 95.43%
Further testing on number of epoch and num of neurons in hidden layers will need to be done
Hyperparameters Tested:
Activation (relu, softmax, sigmoid), Baseline = relu
SGD optimizer Learning Rate (0.01 to 0.05), momentum = 0.9, Baseline = 0.01
Padding(same, valid), baseline = same
Dropout rate in hidden layers(0.4 to 0.8), baseline = 0.5
Batch size( 32, 64, 128, 256), baseline = 64

Accuracy            |  Precision Rates
:-------------------------:|:-------------------------:
![CNN Accuracy](https://github.com/jvaleros/ML_ClassifyShips/blob/main/images/CNN_accuracy.png) |  ![CNN Precision](https://github.com/jvaleros/ML_ClassifyShips/blob/main/images/CNN_precision.png)

### Logistic Regression(LR)
In order to optimize model accuracy, logistic regression was applied after running the data through a pretrained CNN. We chose keras VGG19 network. The keras VGG19 model has been trained with an extensive set of images of size 224x224. In order to use with our model, we need to adapt this model to our input size (80x80)
By running our data through this pre-trained CNN, we were able to extract the relevant features from the data. After this, applying transfer learning we fed this data to the logistic regression model.
The hyperparameters tested where the number of iterations and the solver type for logistic regression.



Accuracy            |  Precision Rates
:-------------------------:|:-------------------------:
![LR Accuracy](https://github.com/jvaleros/ML_ClassifyShips/blob/main/images/LR_accuracy.png) |  ![LR Precision](https://github.com/jvaleros/ML_ClassifyShips/blob/main/images/LR_precision.png)
### SVM
Data: 3,200 training images, and 800 testing images randomly split
Pixel value stored in RGB channel values, 6400 unsigned integers per color to make 80X80 image
Features extracted using pre trained CNN → Into SVM
Sklearn Support Vector Machine - Support Vector Classification implementation
Highest accuracy model on test data had hyperparameters: C(regularization param) = 10, kernel as the radial basis function. 0.998 Accuracy before noise introduction.
Hyperparameters C and Kernel tuned.

Accuracy            |  Precision Rates
:-------------------------:|:-------------------------:
![SVM Accuracy](https://github.com/jvaleros/ML_ClassifyShips/blob/main/images/SVM_accuracy.png) |  ![SVM Precision](https://github.com/jvaleros/ML_ClassifyShips/blob/main/images/SVM_precision.png)

## Conclusion
After fine tuning, it was found that the CNN (accuracy of 97%) performed the best on the noisy data. LOGREG (96% accuracy) was the second most robust model against Gaussian noise in classifying satellite images with a ship or no ship label. Noise injection had an adverse effect on the SVM, which dropped in accuracy to approximately 70%.
## Authors
Max Ramacher \
Sean Tonthat \
Jaime Valero 
