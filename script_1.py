# SCRIPT 1 - INSPECT DATA
'''mysql substring '''

# Import Libraries
import pandas as pd
import mysql.connector
import os
from datetime import datetime
import csv

# Data
data_dir = r'/home/ccirelli2/Desktop/GSU/2019_Spring/ML_Course/Final_Project/Gotham_Cabs/data'
data_file = 'Train.csv'

# Read CSV File
def read_csv_lbl(file_dir, file_name, limit):

    with open(file_dir + '/' + file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')
        Count = 0
        for row in readCSV:
            if Count < limit:
                print(row)
                Count += 1

    return None



read_csv_lbl(data_dir, data_file, 5)






























