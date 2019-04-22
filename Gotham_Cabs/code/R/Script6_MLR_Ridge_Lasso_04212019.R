# MLR - APPLY RIDGE & LASSO

## DOCUMENTATION__________________________________________________________________________
'
1.) Should we regularize our target?
    https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
'


## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(lattice)
library(ggplot2)
library(caret)  # used for parameter tuning
library(glmnet)
library(pls)
library(ISLR)


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


# M1:   TRAIN MULTILINEAR MODEL___________________________________________________
m1.mlr = lm(duration ~ ., data = s1.train)
m1.summary = summary(m1.mlr)
m1.rse = sqrt((sum(m1.summary$residuals^2)) / nrow(s1.train))



# M2:   TRAIN MULTILINEAR MODEL - APPLY RIDGE______________________________________________

# Separate Target & Feature Values
s1_y = s1.50k.nolimits_ran$duration
s1_x = model.matrix(s1.50k.nolimits_ran$duration ~ ., data = s1.50k.nolimits_ran[, 2:11])

s2_y = s2.100k.nolimits_ran$duration
s2_x = model.matrix(s2.100k.nolimits_ran$duration ~ ., data = s2.100k.nolimits_ran[, 2:11])

# Generate Grid Possible Values Lambda
grid = 10^seq(from = 10, to = -2, length = 100)                 #length = desired length of sequence



# M2:   COMPARE COEFFICIENTS USING DIFFERENT LAMBDAS________________________________________
m_cv <- glmnet(s1_x, s1_y, alpha = 1, lambda = grid, standardize = TRUE)

# 25th Lambda
print(paste('25th Lambda =>', m_cv$lambda[25]))           # Value of 25th Lambda
print((coef(m_cv)[,25]))         # Coefficients derived from 25th Lambda

# 75th Lambda
print(paste('75th Lambda =>', m_cv$lambda[75]))           # Value of 75th Lambda
print((coef(m_cv)[,75]))         # Coefficients derived from 75th Lambda



# M3:   TRAIN OPTIMAL MODEL USING 10KFOLD CROSS VALIDATION__________________________________


# Define Function - Train Model------------------------------------------------------------

ridge_cv <- function(X, Y, grid, c_alpha, opt_lambda, c_plot){
  # Train Cross Validation Model
  m_cv <- cv.glmnet(X, Y, alpha = c_alpha, lambda = grid, standardize = TRUE, nfolds = 10)
  # Plot RSE vs Lambda Selection
  if(c_plot == TRUE){
    plot(m_cv, main = "MLR - 10KFOLD USING RIDGE")
  }
  # Get Best Lambda
  cv_lambda = m_cv$lambda.min
  if(opt_lambda == TRUE){
    print(paste('Optimal lambda =>', round(cv_lambda, 2)))
  }
  # Fit Model w/ Best Lambda
  m_optimal <- glmnet(X, Y, alpha = c_alpha, lambda = cv_lambda, standardize = TRUE)
  y_hat_cv <- predict(m_optimal, X)
  model_cv_rse = sqrt(sum((Y - y_hat_cv)^2) / (length(Y) - 2))
  return(model_cv_rse)
}


# Ridge 
ridge_model = ridge_cv(s1_x, s1_y, grid, c_alpha = 0, opt_lambda = TRUE, c_plot = FALSE)

# Lasso
lasso_model = ridge_cv(s1_x, s1_y, grid, c_alpha = 1, opt_lambda = TRUE, c_plot = FALSE)



























