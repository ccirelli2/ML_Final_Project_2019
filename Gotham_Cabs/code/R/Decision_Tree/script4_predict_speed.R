# RANDOM FOREST_____________________________________________________________________

## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(ggplot2)
library(randomForest)
library(ranger)               # Faster implementation of Random Forest
library(tree)
library(ISLR)
library(MASS)
library(caret)                # library that contains the train() function

## IMPORT PERSONAL FUNCTIONS_____________________________________________________________
source('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/code/R/Decision_Tree/module0_random_forest.R')

## CREATE DATASET_________________________________________________________________________
setwd('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/data')
s1.50k.nolimits        = read.csv('sample1_50k.csv')[2:12]                          #[2:12] drop datetime col. 
s2.100k.nolimits       = read.csv('sample1_100k.csv')[2:12]
s3.250k.nolimits       = read.csv('sample1_250k.csv')[2:12]
s4.50k.wlimits         = read.csv('sample2_wlimits_50k.csv')[2:12]
s5.100k.wlimits        = read.csv('sample2_wlimits_100k.csv')[2:12]
s6.250k.wlimits        = read.csv('sample2_wlimits_250k.csv')[2:12]

# RANDOMIZE DATA__________________________________________________________________________
s1.50k.nolimits_ran    = s1.50k.nolimits[sample(nrow(s1.50k.nolimits)),]
s2.100k.nolimits_ran   = s2.100k.nolimits[sample(nrow(s2.100k.nolimits)),]
s3.250k.nolimits_ran   = s3.250k.nolimits[sample(nrow(s3.250k.nolimits)),]
s4.50k.wlimits_ran     = s4.50k.wlimits[sample(nrow(s4.50k.wlimits)), ]
s5.100k.wlimits_ran    = s5.100k.wlimits[sample(nrow(s5.100k.wlimits)), ]
s6.250k.wlimits_ran    = s6.250k.wlimits[sample(nrow(s6.250k.wlimits)), ]


# Drop Duration Feature
s2.100k.nolimits_ran$duration <- NULL
s1.50k.nolimits_ran$duration <- NULL

# Train Model
m0 = ranger(speed ~., data = s2.100k.nolimits_ran, num.trees = 50, mtry = 9, alpha = 0.1, min.node.size = 5)

# Generate OOB RSE
print('Generating OOB RSE')
m0.oob.rse            = round(sqrt(m0$prediction.error),4)
print(paste('OOB RSE => ', m0.oob.rse))
m0$r.squared

# Generate Prediction Using New Sample Data
print('Generating Test Prediction')
m0.predict            = predict(m0, data.test)
# Calculate Test RSE
print('Generating Test RSE')
m0.test.rse           = round(sqrt(sum((data.test$duration - m0.predict$predictions)^2) / (length(m0.predict$predictions)-2)),4)
list.test.rse[Count]  <<- m0.test.rse
print(paste('Test RSE =>', m0.test.rse))
# Increase Count
Count                 <<- Count + 1
# Return Model
print('Model Completed.  Returning model object to user')
print('-----------------------------------------------------------------------------')
















