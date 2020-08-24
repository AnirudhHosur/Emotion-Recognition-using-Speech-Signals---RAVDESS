# Import librires
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # to use operating system dependent functionality
import librosa # to extract speech features
import wave # read and write WAV files
import matplotlib.pyplot as plt # to generate the visualizations

# To calculate accuracy
import operator
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

# MLP Classifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# LSTM Classifier
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop

def extract_mfcc(wav_file_name):
    #This function extracts mfcc features and obtain the mean of each dimension
    #Input : path_to_wav_file
    #Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    
    return mfccs

##### load radvess speech data #####
radvess_speech_labels = [] # to save extracted label/file
ravdess_speech_data = [] # to save extracted features/file
for dirname, _, filenames in os.walk('D:/ML_Projects 2/Ravdess/Audio_Speech_Actors_01-24/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        radvess_speech_labels.append(int(filename[7:8]) - 1) # the index 7 and 8 of the file name represent the emotion label
        wav_file_name = os.path.join(dirname, filename)
        ravdess_speech_data.append(extract_mfcc(wav_file_name)) # extract MFCC features/file
        
print("Finish Loading the Dataset")


### Shuffle the data 
from sklearn.utils import shuffle
ravdess_speech_data, radvess_speech_labels = shuffle(
    ravdess_speech_data, radvess_speech_labels)

# Split the dataset
x_train,x_test,y_train,y_test= train_test_split(
    ravdess_speech_data, radvess_speech_labels, test_size=0.20, random_state=9)

# FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#prediciting the test set results
y_pred = regressor.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)

# APPLYING DECISION TREE ON THE TRAINING DATASET

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = pd.DataFrame(regressor.predict(x_test))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)


""" This will a little time :) """
# FITTING RANDOM FOREST TO THE TRAINING SET

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(x_train, y_train)

# Predicting on test set 
y_pred = pd.DataFrame(regressor.predict(x_test))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)


# FITTING POLYNOMIAL REGRESSION THE THE TRAINING SET 

from sklearn.preprocessing import PolynomialFeatures 
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 
poly_reg = LinearRegression() 
poly_reg.fit(X_poly, y_train) 

# Predicting a new result with Polynomial Regression 
y_pred = poly_reg.predict(poly.fit_transform(x_test)) 
y_pred = pd.DataFrame(y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)





