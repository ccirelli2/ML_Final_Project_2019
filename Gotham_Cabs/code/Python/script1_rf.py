# PURPOSE - TRAIN RANDOM FOREST MODEL
'''

'''
# Import Libraries
import pandas as pd
import os
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

# Import Pesonal Modules
import module1_rf as m1


# Import Data
data_dir = '/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/data'
s1_50k_raw = pd.read_csv(data_dir + '/' + 'sample1_50k.csv')
s2_50k_raw = pd.read_csv(data_dir + '/' + 'sample2_wlimits_50k.csv')




# Define X, Y
def split_xy(data, xy):
    if xy == 'y':
        return data['duration']
    else:
        return data.iloc[:, 2:12]

x = split_xy(s2_50k_raw, 'x')
y = split_xy(s2_50k_raw, 'y').values


# Train Test Split
num_xtrain = int(len(y) * 0.7)
len_data = len(y)
x_train = x.iloc[0: num_xtrain, :]
y_train = y[0: num_xtrain]
x_test  = x.iloc[num_xtrain : len_data, :]
y_test  = y[num_xtrain : len_data]

# Train Model

rf1 = m1.train_RegressionTree(x_train, y_train, x_test, y_test, 'test')
print(rf1)














