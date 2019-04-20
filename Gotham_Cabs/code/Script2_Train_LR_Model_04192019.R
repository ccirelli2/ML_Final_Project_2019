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
                      d.) Polynomials
                      e.) Take log of features & target


  - m2                a.) Multi-linear regression. 
                      b.) Backward & Forward Propogation
                      c.) Ridge & Lasso regression. 

  Code Documentation------------------------------------
  - Create Factors            https://www.guru99.com/r-factor-categorical-continuous.html
  - Stepwise LR Selection     http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/154-stepwise-regression-essentials-in-r/

'


## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(lattice)
library(ggplot2)
library(caret)  # used for parameter tuning

## CREATE DATASET_________________________________________________________________________
setwd('/home/ccirelli2/Desktop/GSU/2019_Spring/ML_Course/2019_GSU_ML_Final_Project/Gotham_Cabs/data')
s1.50k.data        = read.csv('sample1_50k.csv')

# Dataset - No Pre-Processing (no)-------------------------------------------------------
s1.50k.nopp        = s1.50k.data[2:11]                                        # index:     [2:11] drops the datetime feature as we have already covered this in this in the derived features    
set.seed(123)                                                                 # set seed

# Randomize Data
s1.50k.nopp        = s1.50k.nopp[sample(nrow(s1.50k.nopp)),]                  # sample automatically reorders the elements.  We pass the num of rows in orig dataset to sort entire data. 
s1.50k.nopp.sample = 0.7 * nrow(s1.50k.nopp)

# Train / Test Split
s1.50k.nopp.train  = s1.50k.nopp[1:s1.50k.nopp.sample, ]                      # define training set as 70% of number of observations
s1.50k.nopp.test   = s1.50k.nopp[s1.50k.nopp.sample : nrow(s1.50k.nopp),]     # test set 1 - nrow(train)


# Dataset - Create Factors From Continous Variables---------------------------------------
s1.50k.pp = s1.50k.nopp
s1.50k.pp$pickup_x     <- factor(s1.50k.pp$pickup_x)
s1.50k.pp$pickup_y   <- factor(s1.50k.pp$pickup_y) 
s1.50k.pp$dropoff_x   <- factor(s1.50k.pp$dropoff_x) 
s1.50k.pp$dropoff_y   <- factor(s1.50k.pp$dropoff_y) 
s1.50k.pp$weekday   <- factor(s1.50k.pp$weekday, order = TRUE)              # ordinal 
s1.50k.pp$hour_   <- factor(s1.50k.pp$hour_, order = TRUE)                  # ordinal
s1.50k.pp$day_   <- factor(s1.50k.pp$day_, order = TRUE)                    # ordinal
s1.50k.pp$Month_   <- factor(s1.50k.pp$Month_, order = TRUE)                # ordinal
is.factor(s1.50k.pp$weekday)                                                  # Check if a factor
is.ordered((s1.50k.pp$weekday))                                               # Check if ordered

# Train / Test Split
s1.50k.pp        = s1.50k.pp[sample(nrow(s1.50k.pp)),]                  # sample automatically reorders the elements.  
                                                                        # Within the index of our dataframe, we pass sample.   Within sample() we pass the total num of rows. 
s1.50k.pp.sample = 0.7 * nrow(s1.50k.pp)                                # Sample = 70% of all observations. 
s1.50k.pp.train  = s1.50k.pp[1:s1.50k.pp.sample, ]                      # define training set as 1 to (.7 * total of number of observations)
s1.50k.pp.test   = s1.50k.pp[s1.50k.pp.sample : nrow(s1.50k.pp),]     # test set 1 - nrow(train)



## Simple Linear Regression_________________________________________________________


# 1.) Regression On Each Feature - No PreProcessing

# Duration vs Distance
m1.lr = lm(duration ~ distance, data = s1.50k.nopp.train)
summary(m1.lr)
nnnm1.lr.predict = predict(m1.lr, s1.50k.nopp.test)
m1.mse = sqrt(sum(s1.50k.nopp.test$duration - m1.lr.predict)^2) # Calculate Square Root of Residual Sum of Squared Errors. 
m1.mse


# Duration vs All Features
m1.r2 = list()
for (i in names(s1.50k.nopp.train[2:10])){
  m1.train = lm(duration ~ get(i), data = s1.50k.nopp.train)
  m1.summary = summary(m1.train)
  m1.r2[[i]] = round(m1.summary$r.squared,4)
}

m1.r2                                                                          # stdout
setwd('/home/ccirelli2/Desktop/GSU/2019_Spring/ML_Course/2019_GSU_ML_Final_Project/Gotham_Cabs/output')
write.table(m1.r2, 'Simple_lr_model1_r2_train_04192019.xlsx')                        # Write Results to Excel


# 2.) Regression On Each Feature - Features As Factors
m2.lr = lm(duration ~ distance, data = s1.50k.pp.train)
summary(m2.lr)

m2.r2 = list()                                                        # Define List to Catch R2 values

for (i in names(s1.50k.pp.train[2:10])){
  m1.train = lm(duration ~ get(i), data = s1.50k.pp.train)
  m1.summary = summary(m1.train)
  m2.r2[[i]] = round(m1.summary$r.squared, 4)
  
}

m2.r2                                                                 # Print Results
setwd('/home/ccirelli2/Desktop/GSU/2019_Spring/ML_Course/2019_GSU_ML_Final_Project/Gotham_Cabs/output')
write.table(m2.r2, 'Simple_lr_model2_r2_04192019_train.xlsx')                        # Write Results to Excel



## Multilinear Regression____________________________________________________________________________________

# Train Model3 - Multi-linear regression (mlr)
m3.mlr = lm(duration ~ ., data = s1.50k.nopp.train)
m3.summary = summary(m3.mlr)                                            # return summary
m3.summary
#sink('m3_mlr_summary.txt')                                              # divert stout to file
#print(m3.summary)                                                       # call stout


# Train Model4 - Exclude Long / Lat Coordinates
m4.mlr = lm(duration ~ weekday + hour_ + day_ + distance + Month_, data = s1.50k.nopp.train)
m4.summary = summary(m4.mlr)
m4.summary

# Train Model 5: Backward Selection
train.control = trainControl(method = 'cv', number = 10)

m5.backward = train(duration ~ ., data = s1.50k.nopp.train, 
                 method = 'leapBackward',                                 # Step selection
                 tuneGrid = data.frame(nvmax = 1:7),                      # Number of features to consider in the model
                 trControl = train.control)                               # Cross Validation technique 
m5.backward$results
summary(m5.backward$finalModel)      # An asterisk specifies that the feature was included in the model


# Train Model 6:  Forward Selection
m6.forward = train(duration ~ ., data = s1.50k.nopp.train, 
                   method = 'leapForward', 
                   tuneGrid = data.frame(nvmax = 1:7), 
                   trControl = train.control)

m6.forward$results
summary(m6.forward$finalModel)     


# Train Model 7:  Stepwise Selection Using Processed Data
train.control = trainControl(method = 'cv', number = 10)
m7.backward = train(duration ~ ., data = s1.50k.pp.train, 
                  method = 'leapBackward', 
                  tuneGrid = data.frame(nvmax = 1:7), 
                  trControl = train.control)
m7.backward$results
summary(m7.backward$finalModel)     

# How can you make a prediction suing train?


# Need to try Ridge & Lasso








