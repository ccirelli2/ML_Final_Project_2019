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
setwd('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/data')
s1.50k.nolimits        = read.csv('sample1_50k.csv')[2:12]                          #[2:12] drop datetime col. 
s2.100k.nolimits       = read.csv('sample1_100k.csv')[2:12]
s3.250k.nolimits       = read.csv('sample1_250k.csv')[2:12]
s4.50k.wlimits         = read.csv('sample2_wlimits_50k.csv')[2:12]
s5.100k.wlimits        = read.csv('sample2_wlimits_100k.csv')[2:12]
s6.250k.wlimits        = read.csv('sample2_wlimits_250k.csv')[2:12]

# SET SEED FOR ENTIRE CODE________________________________________________________________
set.seed(123)                                                                 


# TRAIN / TEST SPLIT______________________________________________________________________

# Train
s1.train = s1.50k.nolimits[1:  (nrow(s1.50k.nolimits)  * .7), ]
s2.train = s2.100k.nolimits[1: (nrow(s2.100k.nolimits) * .7), ]
s3.train = s3.250k.nolimits[1: (nrow(s3.250k.nolimits) * .7), ]
s4.train = s4.50k.wlimits[1:   (nrow(s4.50k.wlimits)  * .7), ]
s5.train = s5.100k.wlimits[1:  (nrow(s5.100k.wlimits)  * .7), ]
s6.train = s6.250k.wlimits[1:  (nrow(s6.250k.wlimits)  * .7), ]

# Test
s1.test = s1.50k.nolimits[1:  (nrow(s1.50k.nolimits)  * .3), ]
s2.test = s2.100k.nolimits[1: (nrow(s2.100k.nolimits) * .3), ]
s3.test = s3.250k.nolimits[1: (nrow(s3.250k.nolimits) * .3), ]
s4.test = s4.50k.wlimits[1:   (nrow(s4.50k.wlimits)  * .3), ]
s5.test = s5.100k.wlimits[1:  (nrow(s5.100k.wlimits)  * .3), ]
s6.test = s6.250k.wlimits[1:  (nrow(s6.250k.wlimits)  * .3), ]


## M1:   Duration vs Distance_______________________________________________________________________________________
m1.lr            = lm(duration ~ distance, data = s1.train)
m1.summary       = summary(m1.lr)
m1.summary
m1.train.r2      = m1.summary$r.squared
m1.train.ss      = sum(m1.summary$residuals^2) 
m1.train.mse     = mean(m1.summary$residuals^2) 

# M1 Prediction
m1.lr.predict    = predict(m1.lr, s1.test)
m1.ss            = sum((s1.test$duration - m1.lr.predict)^2) # Calculate Square Root of Residual Sum of Squared Errors. 
m1.mse           = mean((s1.test$duration - m1.lr.predict)^2)

# Train Model - s1-3
training_sets <- list(s1.train, s2.train, s3.train, s4.train, s5.train, s6.train)
set.names     <- c('s1.train', 's2.train', 's3.train', 's4.train', 's5.train', 's6.train')
m1.results    <- data.frame('Index' = 1)
Count          = 0

for (i in training_sets){
  lr = lm(duration ~ distance, data = i)
  lr.summary = summary(lr)
  r.squared  = round(lr.summary$r.squared,4)
  Count = Count + 1
  m1.results[Count] <- r.squared
  }
setwd('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/output')
write.csv(m1.results, 'm1_lr_r2_datasets_1to6_04202019.csv')



## M2:  Duration vs Speed____________________________________________________________________________________________

m2.lr            = lm(duration ~ speed, data = s1.train)
m2.summary       = summary(m2.lr)
m2.summary
m2.train.r2      = m2.summary$r.squared
m2.train.ss      = sum(m2.summary$residuals^2) 
m2.train.mse     = mean(m2.summary$residuals^2) 
m2.lr.predict    = predict(m2.lr, s1.test)
m2.ss            = sum((s1.test$duration - m2.lr.predict)^2) # Calculate Square Root of Residual Sum of Squared Errors. 
m2.mse           = mean((s1.test$duration - m2.lr.predict)^2)

# Train Model - s1-3
m2.training.sets <- list(s1.train, s2.train, s3.train, s4.train, s5.train, s6.train)
m2.set.names     <- c('s1.train', 's2.train', 's3.train', 's4.train', 's5.train', 's6.train')
m2.results    <- data.frame('Index' = 1)
Count          = 0

for (i in m2.training.sets){
  lr = lm(duration ~ speed, data = i)
  lr.summary = summary(lr)
  r.squared  = round(lr.summary$r.squared,4)
  Count = Count + 1
  m2.results[Count] <- r.squared
}
setwd('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/output')
write.csv(m2.results, 'm2_lr_r2_datasets_1to6_04202019.csv')









