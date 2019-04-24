# DOCUMENTATION:      FINAL MODEL_____________________________________________________________________________________
'
1.) Data Set
    - Limits:         Since the dataset with limits did better, we will use that one for our final model. 
    - Dummy cols:     Since using dummy columns did better in most tests we will convert all categorical features to dummy cols. 

2.) Feature Selection:
    - Ridge & Lasso   According to Ridge & Lasso the model performed best with between 2-11
                      features performed best.  In all cases distance, speed pickup_x, and 
                      weekday were most important.  We will use backward & forward selection to train
                      the best model. 

3.) Polynomials:
    - Non-linearity   Our models performed much better with the addition of polynomials. 
                      It appears that a polynomail of 2 performed the best, so we will incorporate
                      this into our final model.

4.) PCA:
    - Application     Ultimately, we may want to use PCA to better understand the most important features in our model. 

5.) CV:
    - Application     Cross Validation using 10-Kfold will be used to train our final model. 
                      Should we apply this at the end after training our models?'
      

'THOUGHTS

1.) Maybe try using train() to add the polynomial functions. 
2.) Create dummy factors and only add polynomial ^2 to the continous features.  Then just create lists of features to add to the model
    via forward propogation to see how the model RSE changes. 
    - Pro:  only apply polynomial to continuous features. 
            can use stepwise selection
            can use dummy variables
   - Cons:  Cant use CV. 
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

# CREATE DUMMY VARIABLES__________________________________________________________________

## S1 - Factor
s1.50k.nolimits$pickup_x    <- factor(s1.50k.nolimits$pickup_x)
s1.50k.nolimits$pickup_y    <- factor(s1.50k.nolimits$pickup_y)
s1.50k.nolimits$dropoff_x   <- factor(s1.50k.nolimits$dropoff_x)
s1.50k.nolimits$dropoff_y   <- factor(s1.50k.nolimits$dropoff_y)
s1.50k.nolimits$weekday     <- factor(s1.50k.nolimits$weekday)
s1.50k.nolimits$hour_       <- factor(s1.50k.nolimits$hour_)
s1.50k.nolimits$day_        <- factor(s1.50k.nolimits$day_)
s1.50k.nolimits$month_      <- factor(s1.50k.nolimits$month_)
s1.50k.nolimits             <- dummy_cols(s1.50k.nolimits)              # Create Dummy Columns

s2.100k.nolimits$pickup_x    <- factor(s2.100k.nolimits$pickup_x)
s2.100k.nolimits$pickup_y    <- factor(s2.100k.nolimits$pickup_y)
s2.100k.nolimits$dropoff_x   <- factor(s2.100k.nolimits$dropoff_x)
s2.100k.nolimits$dropoff_y   <- factor(s2.100k.nolimits$dropoff_y)
s2.100k.nolimits$weekday     <- factor(s2.100k.nolimits$weekday)
s2.100k.nolimits$hour_       <- factor(s2.100k.nolimits$hour_)
s2.100k.nolimits$day_        <- factor(s2.100k.nolimits$day_)
s2.100k.nolimits$month_      <- factor(s2.100k.nolimits$month_)
s2.100k.nolimits             <- dummy_cols(s2.100k.nolimits)              # Create Dummy Columns

s4.50k.wlimits$pickup_x     <- factor(s4.50k.wlimits$pickup_x)
s4.50k.wlimits$pickup_y     <- factor(s4.50k.wlimits$pickup_y)
s4.50k.wlimits$dropoff_x    <- factor(s4.50k.wlimits$dropoff_x)
s4.50k.wlimits$dropoff_y    <- factor(s4.50k.wlimits$dropoff_y)
s4.50k.wlimits$weekday      <- factor(s4.50k.wlimits$weekday)
s4.50k.wlimits$hour_        <- factor(s4.50k.wlimits$hour_)
s4.50k.wlimits$day_         <- factor(s4.50k.wlimits$day_)
s4.50k.wlimits$month_       <- factor(s4.50k.wlimits$month_)
s4.50k.wlimits              <- dummy_cols(s4.50k.wlimits)               # Create Dummy Columns

s5.100k.wlimits$pickup_x    <- factor(s5.100k.wlimits$pickup_x)
s5.100k.wlimits$pickup_y    <- factor(s5.100k.wlimits$pickup_y)
s5.100k.wlimits$dropoff_x   <- factor(s5.100k.wlimits$dropoff_x)
s5.100k.wlimits$dropoff_y   <- factor(s5.100k.wlimits$dropoff_y)
s5.100k.wlimits$weekday     <- factor(s5.100k.wlimits$weekday)
s5.100k.wlimits$hour_       <- factor(s5.100k.wlimits$hour_)
s5.100k.wlimits$day_        <- factor(s5.100k.wlimits$day_)
s5.100k.wlimits$month_      <- factor(s5.100k.wlimits$month_)
s5.100k.wlimits             <- dummy_cols(s5.100k.wlimits)              # Create Dummy Columns



# TRAIN / TEST SPLIT______________________________________________________________________

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

# M1    MULTILINEAR REGRESSION_________________________________________
'1.) Use Data w/ Limits
 2.) Use Polynomial of 1-2
 3.) Try backward & forward selection
 4.) Finalize with K-fold. 
'

# Create Training Method - Method = Cross Validation, Folds = 10
train.control = trainControl(method = 'cv', number = 10)









