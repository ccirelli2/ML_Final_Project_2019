# TRAIN LINEAR REGRESSION MODELS
'DOCUMENTATION

  Data-----------------------------------------------
  - Year:             Drop as all values are for 2034
  - Pickup_datetime:  Replace with derived values weekday, hour, day
  - Factors:          Try converting 

  Sample----------------------------------------------
  - Statification:    Do we need to statify our sample based on the target variable?  
  - X & Y Coord:      Do we need to create dummy variables for these?
                      ?? Do we even need the X & Y coordinates if we have duration?

  Approach---------------------------------------------
  - m1                a.) Simply Linear Regression
                      b.) Convert pickup, dropoff, weekday, hour, day to factors. 
                      c.) 10 Kfold CV

  - m2                a.) Multi-linear regression. 
                      b.) Backward & Forward Propogation
                      c.) Ridge & Lasso regression. 

  Code Documentation------------------------------------
  - Create Factors    https://www.guru99.com/r-factor-categorical-continuous.html
'


# M1 - Simple Linear Regression----------------------------------------------------------
















# CLEAR NAMESPACE
rm(list = ls())

# LOAD DATASET
'*Note if you are running this code on your local computer you will need to reset the working-directory'
setwd('/home/ccirelli2/Desktop/GSU/2019_Spring/ML_Course/2019_GSU_ML_Final_Project/Gotham_Cabs/data')
sample1 = read.csv('sample1_100k.csv')
sample1.drop.date = head(sample1[2:10])


