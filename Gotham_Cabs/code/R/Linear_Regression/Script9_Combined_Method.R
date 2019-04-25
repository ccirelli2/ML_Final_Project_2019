# FINAL MODEL


## READINGS
'

https://stackoverflow.com/questions/51020700/ridge-regression-with-glmnet-for-polynomial-and-interactions-terms-in-r'



## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(lattice)
library(ggplot2)
library(caret)  # used for parameter tuning
library(glmnet)
library(pls)
library(ISLR)
library(boot)
library(fastDummies)

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

# Separate Target & Feature Values
s1_y = s1.50k.nolimits_ran$duration
s1_x = as.matrix(s1.50k.nolimits_ran[,2:11])

s2_y = s2.100k.nolimits_ran$duration
s2_x = as.matrix(s2.100k.nolimits_ran[,2:11])

s3_y = s3.250k.nolimits_ran$duration
s3_x = as.matrix(s3.250k.nolimits_ran[,2:11])

s4_y = s4.50k.wlimits_ran$duration
s4_x = as.matrix(s4.50k.wlimits_ran[,2:11])

s5_y = s5.100k.wlimits_ran$duration
s5_x = as.matrix(s5.100k.wlimits_ran[,2:11])

s6_y = s6.250k.wlimits_ran$duration
s6_x = as.matrix(s6.250k.wlimits_ran[,2:11])

# Generate Grid Possible Values Lambda
grid = 10^seq(from = 10, to = -2, length = 100)                 #length = desired length of sequence


# M1:   TRAIN RIDGE & LASSO MODELS - COMPARE LAMBDAS__________________________________________

for (i in seq(2, 3.5, .1)){
  print(paste('Training Model w/ Polynomial =>', i))
  x <- model.matrix(duration ~ poly(pickup_x, pickup_y, dropoff_x, dropoff_y, 
                                    weekday, hour_,  day_,distance, month_, speed, 
                                    degree = i, raw = T), data = s6.250k.wlimits_ran)
  y = s6.250k.wlimits_ran$duration
  m_cv = cv.glmnet(x, y, alpha = 0, lambda = grid, standardize = TRUE, nfolds = 10)
  cv_lambda = m_cv$lambda.min
  m_optimal <- glmnet(x, y, alpha = 0, lambda = cv_lambda, standardize = TRUE)
  y_hat_cv <- predict(m_optimal, x)
  model_cv_rse = sqrt(sum((y - y_hat_cv)^2) / (length(Y) - 2))
  print(paste('Poly => ', i, ' TEST RSE => ', round(model_cv_rse ,4)))
}



# Create DataFrame
df = data.frame(row.names = names.datasets)
df$ridge.rse = rse.test


