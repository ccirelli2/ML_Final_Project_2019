# Import Libraries
import pandas as pd
import os
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV





def train_RegressionTree(x_train, y_train, x_test, y_test, tt):

    # Instantiate the RF Model
    clf_RF = DecisionTreeRegressor(max_depth = 10)
              #min_samples_leaf  = 2,
              #max_features      = 'auto',
              #max_depth         = 50,
              #bootstrap         = False)
                                    #)
    # Train the Model
    clf_RF.fit(x_train, y_train)
    # Generate & Prediction On X_test data
    y_predict = clf_RF.predict(x_test)

    if tt == 'train':
        return accuracy_score(y_train, clf_RF.predict(x_train))

    elif tt == 'test':
        return accuracy_score(y_test, y_predict)

    # Return
    return None

