# Load Standard Libraries
import pandas as pd
import os
import mysql.connector
from datetime import datetime
import matplotlib.pyplot as plt

# Import Personal Modules


# Load Scikit Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load Keras Libraries
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout, Activation



# FUNCTIONS________________________________________________________________

def split_xy(data, xy):
    if xy == 'y':
        return data['duration']
    else:
        return data.iloc[:, 2:12]

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()




# MODELS_____________________________________________________________________

def cnn_model(x_train, y_train, x_test, y_test, input_dim, num_layers = 1,
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
        model.add(layers.Dense(units = 1, input_dim = input_dim, activation = 'relu'))

    elif num_layers ==2:
        model.add(layers.Dense(units = 18, input_dim = input_dim, activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(layers.Dense(units = 9, activation = 'softmax'))

    elif num_layers == 3:
        model.add(layers.Dense(units = 32, input_dim = input_dim, activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(layers.Dense(units = 18))
        model.add(layers.Dense(units = 9, activation = 'softmax'))

    elif num_layers == 4:
        model.add(layers.Dense(units = 64, input_dim = input_dim, activation = 'relu'))
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






