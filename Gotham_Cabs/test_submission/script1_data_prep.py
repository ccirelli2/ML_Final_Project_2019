# Prep Target Dataset


# Import Libraries
import os
import pandas as pd
from datetime import datetime
from math import sqrt

# Data Location
data_wd = '/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/test_submission'
file_name= 'Test.csv'

# Import File as a Dataframe
df = pd.read_csv(data_wd + '/' +  file_name)

# Convert Features
df['pickup_x']  =  [int(x) for x in df['pickup_x']]       
df['pickup_y']  =  [int(x) for x in df['pickup_y']]
df['dropoff_x'] =  [int(x) for x in df['dropoff_x']]
df['dropoff_y'] =  [int(x) for x in df['dropoff_y']]
df['month']     =  [datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month for x in df['pickup_datetime']]
df['day']       =  [datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day for x in df['pickup_datetime']]
df['hour']      =  [datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour for x in df['pickup_datetime']]
df['weekday']   =  [datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday() for x in df['pickup_datetime']]


def calc_distance(df):
    list_distance = []

    for row in df.itertuples():
        # Define Features
        pickup_x = int(row[2])
        pickup_y = int(row[3])
        dropoff_x = int(row[4])
        dropoff_y = int(row[5])      
        # Calculate Distance
        a = (dropoff_x - pickup_x) * (dropoff_x - pickup_x)
        b = (dropoff_y - pickup_y) * (dropoff_y - pickup_y)
        c = round(sqrt(a + b),0)
        list_distance.append(c)

    df['distance'] = [int(x) for x in list_distance]

    return df

# Run Function
df = calc_distance(df)
print(df.head())


# Write DataFrame to Local Drive
df.to_csv('Final_test_data_clean_version.csv')



