# Prep Target Dataset


# Import Libraries
import os
import pandas as pd
from datetime import datetime

# Data Location
data_wd = '/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/test_submission'
file_name= 'Test.csv'

# Import File as a Dataframe
df = pd.read_csv(data_wd + '/' +  file_name)

df['pickup_x']  =  [int(x) for x in df['pickup_x']]
df['pickup_y']  =  [int(x) for x in df['pickup_y']]
df['dropoff_x'] =  [int(x) for x in df['dropoff_x']]
df['dropoff_y'] =  [int(x) for x in df['dropoff_y']]
df['month']     =  [datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month for x in df['pickup_datetime']]








