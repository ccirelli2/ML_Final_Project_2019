



# LIBRARIES________________________________________________________________
# Load Standard Libraries
import pandas as pd
import os
import mysql.connector
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


# Import Personal Modules
import module2_cnn as m2

# Load Scikit Learn Libraries
from sklearn.model_selection import train_test_split


# Load Keras Libraries
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout, Activation

# Dataset:
data_dir = '/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/data'
s1_100k_raw = pd.read_csv(data_dir + '/' + 'sample1_100k.csv')
s2_100k_raw = pd.read_csv(data_dir + '/' + 'sample2_wlimits_100k.csv')

# Split X & Y
'''
x_split = m2.split_xy(s2_100k_raw, 'x')
y_split = m2.split_xy(s2_100k_raw, 'y')
'''

x    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
y    = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)


# Create Dummy Cols 
'''
dum_cols = ['pickup_x', 'pickup_y', 'dropoff_x','dropoff_y', 'weekday', 'hour_', 
          'day_', 'month_']
x_dums = pd.get_dummies(x_raw, columns = dum_cols)

x = x_dums.values
y = y_raw.values
'''

# Split Train / Test
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.30, random_state = 1000)

# Input Dim
input_dim = [1]#x.shape[1]

# Train Model
'''m2.cnn_model(x_train, y_train, x_test, y_test, input_dim, num_layers = 1, 
             loss='mse', optimizer='adam', num_epochs=10, n_batch_size=10, 
             model_summary=True, plot=True)
'''

model = Sequential()
model.add(layers.Dense(units = 1, input_shape= [1], activation = 'relu'))
model.compile(loss='mse', optimizer='adam', 
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs= 10,
                    verbose=False,
                    validation_data=(x_test, y_test))
# Measure Accuracy of Model
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print('Training Accuracy: {}'.format(round(accuracy,4)))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print('Test Accuracy: {}'.format(round(accuracy, 4)))

m2.plot_history(history)



















