# Load Standard Libraries
import pandas as pd
import os
import mysql.connector
from datetime import datetime
import matplotlib.pyplot as plt
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
s1_100k  = pd.read_csv(data_dir + '/' + 'sample2_wlimits_100k.csv')
cols     = ['pickup_x', 'pickup_y', 'dropoff_x', 'dropoff_y', 'weekday', 
            'hour_', 'day_', 'distance', 'month_', 'speed']

x = s1_100k[cols]
y = s1_100k['duration']

#x    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
#y    = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

train_num = int(len(x) * .7)
x_train   = x[0:train_num]
y_train   = y[0:train_num]
x_test    = x[train_num:]
y_test    = y[train_num:]

num_features = len(x.columns)


l0 = tf.keras.layers.Dense(units=1, input_shape=[num_features], activation='relu')
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01), 
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=15, verbose=False, 
                    validation_data=(x_test, y_test))

# Measure Accuracy of Model

loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
train_rse = round(math.sqrt(loss),4)
print('Train Accuracy => {}'.format(accuracy))
print('Train RSE      => {}'.format(train_rse))
loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
test_rse = round(math.sqrt(loss),4)
print('Test RSE       => {}'.format(test_rse))
print('Test Accuracy  => {}'.format(accuracy))

'''
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

plot_history(history)
plt.show()
'''
