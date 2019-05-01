# GSU - DEEP LEARNING - PROJECT B
'''Readings
    https://www.tensorflow.org/guide/keras

'''

# LIBRARIES________________________________________________________________
# Load Standard Libraries
import pandas as pd
import os
import mysql.connector
from datetime import datetime
import matplotlib.pyplot as plt

# Import Personal Modules
import module1 as m1

# Load Scikit Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load Keras Libraries
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout, Activation


# DATA_____________________________________________________________________
# Create Connection to MySQL DB
mydb = mysql.connector.connect(
        host="localhost",
        user="ccirelli2",
        passwd="Work4starr",
        database='GSU'
        )
# Load Data
df = m1.get_labeled_data(mydb)


# DATA PREP____________________________________________________________
# Split X & Y
y_raw       = df['LABEL'].values
y           = pd.get_dummies(y_raw).values                 # convert y to dummy cols
sentences   = df['TEXT'].values
# Split Train & Test
sentences_train, sentences_test, y_train, y_test = train_test_split(
                    sentences, y, test_size=0.30, random_state=1000)
# Vectorize
vectorizer = CountVectorizer(min_df = 3, lowercase = True, stop_words = 'english', 
                             max_features = 10000)
vectorizer.fit(sentences_train)
x_train = vectorizer.transform(sentences_train).toarray()
x_test  = vectorizer.transform(sentences_test).toarray()
# Get Num of Features 
num_features = 9080 




# MODEL_____________________________________________________________________

def CNN_model(x_train, y_train, x_test, y_test, num_features, num_layers = 1, 
              loss='binary_crossentropy', optimizer ='adam', num_epochs=25, 
              n_batch_size = 500, model_summary = True, plot = True):

    '''Inputs
    1. num_layers:      Choose number of layers to trian CNN
    2. loss:            See keras documentation for full list of options https://keras.io/losses/
    3. optimizer        See keras https://keras.io/optimizers/
    4. num_epochs:      Number of epochs to use in training
    5. n_batch_size:    Sample size to use for each node 
    6. plot:            If you want to plot the Train/Test accuracy and loss curves
    '''

    # Instantiate Model
    model = Sequential()

    # Add Layers
    '''input_shape: specifies the dimension of the input to the layer
        activation:  function used to activate the layer
        units:       number of neurons in the layer.  if our output is 1, and we are on the last
                layer, then units should = 1.'''

    # Build Model
    if num_layers ==1:
        model.add(layers.Dense(units = 9, input_shape = [num_features], activation = 'relu'))

    elif num_layers ==2:
        model.add(layers.Dense(units = 18, input_shape = [num_features], activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(layers.Dense(units = 9, activation = 'softmax'))

    elif num_layers == 3:
        model.add(layers.Dense(units = 32, input_shape = [num_features], activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(layers.Dense(units = 18))
        model.add(layers.Dense(units = 9, activation = 'softmax'))

    elif num_layers == 4:
        model.add(layers.Dense(units = 64, input_shape = [num_features], activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(layers.Dense(units = 32))
        model.add(layers.Dense(units = 18))
        model.add(layers.Dense(units = 9, activation = 'softmax'))
            
    # Specify Optomizer
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    # Prints Summary of NN Structure
    if model_summary == True:
        model.summary()

    # Set Number of Epochs
    history = model.fit(x_train, y_train, 
                    epochs= num_epochs, 
                    verbose=False, 
                    validation_data=(x_test, y_test),
                    batch_size=n_batch_size)

    # Measure Accuracy of Model
    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print('Training Accuracy: {}'.format(round(accuracy,4)))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print('Test Accuracy: {}'.format(round(accuracy, 4)))

    if plot == True:
        plt.style.use('ggplot')
        m1.plot_history(history)
        plt.show()
    
    # End Of Function---------------------------------



CNN_model(x_train, y_train, x_test, y_test, num_layers = 4, 
          num_features = num_features, loss='binary_crossentropy', 
          num_epochs = 15, n_batch_size = 500, 
          model_summary = True, plot = True)





