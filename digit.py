# for digit recognizer kaggle competition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearnmodel_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categroical # convert to one-hot encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2d
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', paletter='deep')

# Load the data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

Y_train = train["label"]

# Drop "label" column
X_train = train.drop(labels = ["label"].axis = 1)

# free some space
del train

Y_train.value_counts()

# Check the date
X_train.isnull().any().describe()

test.isnull().any().describe()

#Checking for corrupted images
#There is nomissing values in the train
#and test dataset.

#Normalize the data.
X_train = X_train/255.0
test = test /255.0

# Reshape image in 3 dimension 
# (height = 28px, width = 28px, canal = 1)

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

#Train and test images (28px x 28px) has been put into pandas.Dataframes
#as 1D vectors of 784 values.
#we reshape as 28x28x1 3D matrices.

#Keras requires an extra dimension in the end which corresponds to channels.
#MNIST images are gray scaled so it only uses one channel.
#For RGB images, there is 3 channels, so we would have had matrices 28x28x3 if colored.

#Encode labels to one hot vectors
#(ex : 2 -> [0,0,1,0,0,0,0,0,0,0]
Y_train = to_categorical(Y_train, num_classes = 10)

#Labels are 10 digit numbers from 0 to 9.
#We need to encode these labels to one hot vectors
#(ex: 2 -> [0,0,1,0,0,0,0,0,0,0,0])

#set the random seed
random_seed = 2

#Split the train and the validation set for fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = random_seed)


