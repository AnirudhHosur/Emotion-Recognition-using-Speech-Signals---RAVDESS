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

"""
### Shuffle the data 
from sklearn.utils import shuffle
ravdess_speech_data, radvess_speech_labels = shuffle(
    ravdess_speech_data, radvess_speech_labels) """

# PREPARE THE DATASET FOR ML MODELS
#### convert data and label to array
ravdess_speech_data_array = np.asarray(
    ravdess_speech_data) # convert the input to an array
ravdess_speech_label_array = np.array(radvess_speech_labels)
ravdess_speech_label_array.shape # get tuple of array dimensions

#### make categorical labels
# converts a class vector (integers) to binary class matrix
labels_categorical = to_categorical(ravdess_speech_label_array) 
ravdess_speech_data_array.shape
labels_categorical.shape

""" STEP 1 - MLP CLASSIFIER """
x_train,x_test,y_train,y_test= train_test_split(
    np.array(ravdess_speech_data_array),
    labels_categorical, test_size=0.15, random_state=9)

"""
# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(
    alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
    learning_rate='adaptive', max_iter=400,shuffle=True)
# Train the model
model.fit(x_train,y_train)
# Predict for the test set
y_pred=model.predict(x_test)
# Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100)) """


### MLP Classifier using Sckit 
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=100)
# Define hyperparamter space to search
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
# Run the search
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(x_train, y_train)
# See the best results
# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = y_test , clf.predict(x_test)

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))
rmse = np.sqrt(mean_squared_error(y_true,y_pred))
r2 = r2_score(y_true,y_pred)
print(rmse)
print(r2)








