# PURPOSE - TRAIN CONVOLUTIONAL NEURAL NETWORK
'''
    Keras metrics: https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
'''
# Load Standard Libraries
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math

# Import Personal Modules
import module2_cnn as m2

# Load Scikit Learn Libraries
from sklearn.model_selection import train_test_split

# Load Keras Libraries
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout, Activation

# Dataset
data_dir = '/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/data'
s1_50k_nl  = pd.read_csv(data_dir + '/' + 'sample1_50k.csv')
s2_100k_nl = pd.read_csv(data_dir + '/' + 'sample1_100k.csv')
s3_150k_nl = pd.read_csv(data_dir + '/' + 'sample1_250k.csv')
s4_50k_wl  = pd.read_csv(data_dir + '/' + 'sample2_wlimits_50k.csv')
s5_150k_wl = pd.read_csv(data_dir + '/' + 'sample2_wlimits_100k.csv')
s6_250k_wl = pd.read_csv(data_dir + '/' + 'sample2_wlimits_250k.csv')

# Define Columns TO use in Function
cols     = ['pickup_x', 'pickup_y', 'dropoff_x', 'dropoff_y', 'weekday', 
            'hour_', 'day_', 'distance', 'month_', 'speed']
x = s6_250k_wl[cols]
y = s6_250k_wl['duration']


# Define Train Size & Generate Train/Test Split
train_num = int(len(x) * .7)
x_train   = x[0:train_num]
y_train   = y[0:train_num]
x_test    = x[train_num:]
y_test    = y[train_num:]

# Calculate Number of Features in your x feature set
num_features = len(x.columns)


# BUILD CNN------------------------------------------------------------------------

# Create Layers
'Intention is to try diff combinations of these layers'
l0 = tf.keras.layers.Dense(units=1, input_shape=[num_features], activation='relu')
l1 = tf.keras.layers.Dense(units=10, input_shape=[num_features], activation='relu')
l2 = tf.keras.layers.Dense(units=5, input_shape=[num_features], activation='relu')
l3 = tf.keras.layers.Dense(units=1, input_shape=[num_features], activation='sigmoid')
l4 = tf.keras.layers.Dense(units=30, input_shape=[num_features], activation='relu')
l5 = tf.keras.layers.Dropout(0.25)

# Build Model
'pass to the model the layers you want to use to train it'
model = tf.keras.Sequential([l1, l0])

# Compile Model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01), 
              metrics=['mse'])
model.summary()

# Fit Model
history = model.fit(x_train, y_train, epochs=5, verbose=False, 
                    validation_data=(x_test, y_test))

# Measure Accuracy of Model-------------------------------------

# Training Results
loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
train_rse = round(math.sqrt(loss),4)
print('Train Accuracy => {}'.format(accuracy))
print('Train RSE      => {}'.format(train_rse))
# Test Results
loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
test_rse = round(math.sqrt(loss),4)
print('Test Accuracy  => {}'.format(accuracy))
print('Test RSE       => {}'.format(test_rse))


# Create Plot of Epoch Results
train_loss = history.history['loss']
val_loss   = history.history['val_loss']

# Get Length of arrays
len_train_loss = range(1,len(train_loss)+1)
len_test_loss  = range(1,len(val_loss)+1)

# Generate Plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(len_train_loss, train_loss, 'b', label='Training Loss')
plt.plot(len_test_loss, val_loss, 'r', label='Validation Loss')
plt.title('TRAINING & TEST RSE BY EPOCH')
plt.legend()
plt.show()





