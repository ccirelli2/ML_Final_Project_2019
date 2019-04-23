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

# Calculate Number of Training Observations
train_nrows_50k  = (nrow(s1.50k.nolimits)  * .7)
train_nrows_100k = (nrow(s2.100k.nolimits)   * .7)
train_nrows_250k = (nrow(s3.250k.nolimits)   * .7)

# Train
s1.train = s1.50k.nolimits_ran[1:  (nrow(s1.50k.nolimits_ran)  * .7), ]
s2.train = s2.100k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s3.train = s3.250k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s4.train = s4.50k.wlimits_ran[1:   (nrow(s4.50k.wlimits_ran)  * .7), ]
s5.train = s5.100k.wlimits_ran[1:  (nrow(s5.100k.wlimits_ran)  * .7), ]
s6.train = s6.250k.wlimits_ran[1:  (nrow(s6.250k.wlimits_ran)  * .7), ]

# Test
s1.test = s1.50k.nolimits_ran[train_nrows_50k:    nrow(s1.50k.nolimits_ran), ] # Index from training to total
s2.test = s2.100k.nolimits_ran[train_nrows_100k:  nrow(s2.100k.nolimits_ran), ]
s3.test = s3.250k.nolimits_ran[train_nrows_250k:  nrow(s3.250k.nolimits_ran), ]
s4.test = s4.50k.wlimits_ran[train_nrows_50k:     nrow(s4.50k.wlimits_ran), ]
s5.test = s5.100k.wlimits_ran[train_nrows_100k:   nrow(s5.100k.wlimits_ran), ]
s6.test = s6.250k.wlimits_ran[train_nrows_250k:   nrow(s6.250k.wlimits_ran), ]



# M1:   TRAIN MULTILINEAR MODEL___________________________________________________
m1.mlr = lm(duration ~ ., data = s1.train)
m1.summary = summary(m1.mlr)
m1.rse = sqrt((sum(m1.summary$residuals^2)) / nrow(s1.train))
m1.summary
m1.sqrt.sq.coeff = sum(sqrt(m1.summary$coefficients[,1]^2))                   # square coefficients, take square root and then sum. 
m1.sqrt.sq.coeff                                                              # Compare to coeffs from Ridge & Lasso


# M2:   TRAIN MULTILINEAR MODEL - APPLY RIDGE______________________________________________

# Separate Target & Feature Values
s1_y = s1.train$duration
s1_x = as.matrix(s1.train[,2:11])

s2_y = s4.train$duration
s2_x = as.matrix(s4.train[,2:11])

s3_y = s2.train$duration
s3_x = as.matrix(s2.train[,2:11])

s4_y = s5.train$duration
s4_x = as.matrix(s5.train[,2:11])

# Generate Grid Possible Values Lambda
grid = 10^seq(from = 10, to = -2, length = 100)                 #length = desired length of sequence

# Train MLR Using All Lambdas Grid 
m2.ridge <- glmnet(s1_x, s1_y, alpha = 0, lambda = grid, standardize = TRUE)

# 25th Lambda
print(paste('25th Lambda =>', m2.ridge$lambda[25]))           # Value of 25th Lambda
print((coef(m2.ridge)[,25]))         # Coefficients derived from 25th Lambda
m2.coeff_25th = coef(m2.ridge)[,25]
m2.sqrt.sq.coeff.25th = sum(sqrt(m2.coeff_25th^2))
m2.sqrt.sq.coeff

# 75th Lambda
print(paste('75th Lambda =>', m_cv$lambda[50]))           # Value of 75th Lambda
print((coef(m_cv)[,75]))         # Coefficients derived from 75th Lambda
m2.coeff_75th = coef(m2.ridge)[,50]
m2.sqrt.sq.coeff.75th = sum(sqrt(m2.coeff_75th^2))
m2.sqrt.sq.coeff.75th                                       # Compare to show how coefficients are approaching zero.


# M3:   TRAIN MULTILINEAR MODEL - INSPECT COEFFICEINTS FOR LASSO______________________________

# Train MLR Using All Lambdas Grid 
m3.lasso <- glmnet(s1_x, s1_y, alpha = 1, lambda = grid, standardize = TRUE)

# 25th Lambda
print(paste('25th Lambda =>', m3.lasso$lambda[25]))           # Value of 25th Lambda
print((coef(m3.lasso)[,25]))                                  # Odd, all coefficeints are zero
m3.coeff_25th = coef(m3.lasso)[,25]
m3.sqrt.sq.coeff.25th = sum(sqrt(m3.coeff_25th^2))
m3.sqrt.sq.coeff.25th

# 75th Lambda
print(paste('50th Lambda =>', m_cv$lambda[50]))           # Value of 75th Lambda
print((coef(m3.lasso)[,50]))         # Coefficients derived from 75th Lambda
m3.coeff_50th = coef(m3.lasso)[,50]
m3.sqrt.sq.coeff.50th = sum(sqrt(m3.coeff_50th^2))
m3.sqrt.sq.coeff.50th



# M4:   FIND OPTIMAL LAMBDAS FOR RIDGE & LASSO________________________________________________

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


# List Datasets

# Ridge 
ridge_model = ridge_cv(s4_x, s4_y, grid, c_alpha = 0, opt_lambda = TRUE, c_plot = FALSE)
ridge_model


  # Lasso
lasso_model = ridge_cv(s4_x, s4_y, grid, c_alpha = 1, opt_lambda = TRUE, c_plot = FALSE)
lasso_model


## Results_______________________________________________________________________________
model.name = list('m1.mlr.s1', 'm2.ridge.s1.50k.nl', 'm3.ridge.s2.50k.wl', 'm4.ridge.s4.100k.wl', 'm5.lasso.s1.50k.nl',
                  'm6.lasso.s2.50k.wl', 'm7.lasso.s4.100k.wl')
model.rse = c(280.2, 280.1613, 246.52, 241.053, 280.1571, 246.52, 241.053)


barplot(model.rse,  names.arg = model.name, cex.names = .75, las = 2, 
        main = "Comparison Ridge & Lasso RSE", 
        ylab = 'RSE')
















