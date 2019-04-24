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

# RANDOMIZE DATA__________________________________________________________________________
s1.50k.nolimits_ran    = s1.50k.nolimits[sample(nrow(s1.50k.nolimits)),]
s2.100k.nolimits_ran   = s2.100k.nolimits[sample(nrow(s2.100k.nolimits)),]
s3.250k.nolimits_ran   = s3.250k.nolimits[sample(nrow(s3.250k.nolimits)),]
s4.50k.wlimits_ran     = s4.50k.wlimits[sample(nrow(s4.50k.wlimits)), ]
s5.100k.wlimits_ran    = s5.100k.wlimits[sample(nrow(s5.100k.wlimits)), ]
s6.250k.wlimits_ran    = s6.250k.wlimits[sample(nrow(s6.250k.wlimits)), ]

# TRAIN / TEST SPLIT______________________________________________________________________

# Training Set Sizes
train_nrows_50k  = (nrow(s1.50k.nolimits)  * .7)
train_nrows_100k = (nrow(s2.100k.nolimits)   * .7)
train_nrows_250k = (nrow(s3.250k.nolimits)   * .7)

# Train
s1.train = s1.50k.nolimits_ran[1:  (nrow(s1.50k.nolimits_ran)  * .7), ]
s3.train = s2.100k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s5.train = s3.250k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s2.train = s4.50k.wlimits_ran[1:   (nrow(s4.50k.wlimits_ran)  * .7), ]
s4.train = s5.100k.wlimits_ran[1:  (nrow(s5.100k.wlimits_ran)  * .7), ]
s6.train = s6.250k.wlimits_ran[1:  (nrow(s6.250k.wlimits_ran)  * .7), ]

# Test
s1.test = s1.50k.nolimits_ran[ train_nrows_50k:   length(s1.50k.nolimits_ran), ]
s3.test = s2.100k.nolimits_ran[train_nrows_100k: length(s2.100k.nolimits_ran) ,]
s5.test = s3.250k.nolimits_ran[train_nrows_250k: length(s3.250k.nolimits_ran), ]
s2.test = s4.50k.wlimits_ran[  train_nrows_50k:   length(s1.50k.nolimits_ran), ]
s4.test = s5.100k.wlimits_ran[ train_nrows_100k: length(s2.100k.nolimits_ran) ,]
s6.test = s6.250k.wlimits_ran[ train_nrows_250k: length(s2.100k.nolimits_ran) ,]


## M1:   Duration vs Distance_______________________________________________________________________________________
m1.lr            = lm(duration ~ distance, data = s6.train)
m1.summary       = summary(m1.lr)
m1.summary
m1.train.r2      = m1.summary$r.squared
m1.train.ss      = sum(m1.summary$residuals^2) 
m1.train.mse     = mean(m1.summary$residuals^2) 

# M1 Prediction
m1.lr.predict    = predict(m1.lr, s6.test)
m1.rse           = sqrt(sum((s6.test$duration - m1.lr.predict)^2) / (length(m1.lr.predict) -2))
m1.rse


# GET RSE ALL DATASETS
training_sets <- list(s1.train, s2.train, s3.train, s4.train, s5.train, s6.train)
test_sets     <- list(s1.test, s2.test, s3.test, s4.test, s5.test, s6.test)

index.rse = c()
list.train.rse = c()
list.test.rse  = c()
index_count = 1


for (i in seq(1, length(training_sets))){
  # Start Index 
  index.rse[index_count] = i
  print(paste('training model for dataset =>', i))
  # Create Dataset Objects
  data_train <- training_sets[[i]]
  data_test  <- test_sets[[i]]
  # Train LR Model
  ml.train = lm(duration ~ distance, data = data_train)
  ml.train.summary = summary(ml.train)
  list.train.rse[index_count] = sqrt(sum(ml.train.summary$residuals^2) / length(ml.train.summary$residuals))  
  # Generate A Prediction & Calculate RSE
  print(paste('creating test results for dataset =>', i))
  ml.predict = predict(ml.train, data_test)
  list.test.rse[index_count] = sqrt(sum((data_test$duration - ml.predict)^2) / (length(ml.predict) -2))
  index_count = index_count + 1
}

print('hello world')

# Create DataFrame
df = data.frame(row.names = index.rse)
df$index.rse = index.rse
df$train.rse = list.train.rse
df$test.rse = list.test.rse


# Generate a Plot for Train & Test Points
barplot(data = df, height = df$test.rse, main = 'Simple Linear Reggresion - Compare Datasets', 
        legend = test_sets, beside = TRUE, col = 'darkblue', 
        xlab = 'Datasets', ylab = 'RSE')
 




df






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









