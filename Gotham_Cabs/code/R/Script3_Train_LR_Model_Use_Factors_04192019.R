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
s1.50k.nolimits        = read.csv('sample1_50k.csv')[2:11]
s2.100k.nolimits       = read.csv('sample1_100k.csv')[2:11]
s3.250k.nolimits       = read.csv('sample1_250k.csv')[2:11]
s4.100k.wlimits        = read.csv('sample4_wlimits_100k.csv')[2:11]
s5.250k.wlimits        = read.csv('sample4_wlimits_250k.csv')[2:11]

# SET SEED FOR ENTIRE CODE________________________________________________________________
set.seed(123)                                                                 


# TRAIN / TEST SPLIT______________________________________________________________________

# Train
s1.train = s1.50k.nolimits[1:  (nrow(s1.50k.nolimits)  * .7), ]
s2.train = s2.100k.nolimits[1: (nrow(s2.100k.nolimits) * .7), ]
s3.train = s3.250k.nolimits[1: (nrow(s3.250k.nolimits) * .7), ]
s4.train = s4.100k.wlimits[1:  (nrow(s4.100k.wlimits)  * .7), ]
s5.train = s5.250k.wlimits[1:  (nrow(s5.250k.wlimits)  * .7), ]

# Test
s1.test = s1.50k.nolimits[1:  (nrow(s1.50k.nolimits)  * .3), ]
s2.test = s2.100k.nolimits[1: (nrow(s2.100k.nolimits) * .3), ]
s3.test = s3.250k.nolimits[1: (nrow(s3.250k.nolimits) * .3), ]
s4.test = s4.100k.wlimits[1:  (nrow(s4.100k.wlimits)  * .3), ]
s5.test = s5.250k.wlimits[1:  (nrow(s5.250k.wlimits)  * .3), ]


# Dataset_2 - Create Factors From Continous Variables---------------------------------------
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









