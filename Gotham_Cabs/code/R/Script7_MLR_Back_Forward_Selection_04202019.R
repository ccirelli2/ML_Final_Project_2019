## READING MATERIALS______________________________________________________________________
'http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/154-stepwise-regression-essentials-in-r/'

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



# M1:     BACKWARD SELECTION_____________________________________________________________

# Create Training Method - CV & 10kfold
train.control = trainControl(method = 'cv', number = 10)

# Train Model 
m1.backward = train(duration ~ ., data = s1.50k.nolimits_ran, 
                    method = 'leapBackward',                                 # Step selection
                    tuneGrid = data.frame(nvmax = 1:11),                     # Number of features to consider in the model
                    trControl = train.control)                               # Cross Validation technique 
m1.backward$results
summary(m1.backward$finalModel)      # An asterisk specifies that the feature was included in the model


model_opt <- function(dataset, opt_method, num_param, cv_grid, result2return){
  'opt_method:        either leapBackward or leapForward
   results:           A dataframe with the training error rate and values for the tuning params
   bestTune           A dataframe with the final parameters
   finalModel         Aa fit object using the best parameters'
  # Train Model
  m0 = train(duration ~ ., data = dataset, 
             method    = opt_method,                                   # Step selection
             tuneGrid  = data.frame(nvmax = 1 : num_param),            # Number of features to consider in the model
             trControl = cv_grid)                                      # Cross Validation technique 
  # Results
  if (result2return == 'results'){
    print(paste('Results:',m0$results))
    return(m0$results)}
    
    # Final Model - # An asterisk specifies that the feature was included in the model
    else if (result2return == 'finalModel'){
      print(paste('Final Model:',summary(m0$finalModel)))
      return(summary(m1.backward$finalModel))}      
    
    else if (result2return == 'bestTune'){
      print(paste('Best Tune:', m0$bestTune))
      return(m0$bestTune)}
}


model_opt(s1.50k.nolimits_ran, 'leapBackward', 11, train.control, 'results')


# Train Model 6:  Forward Selection
m6.forward = train(duration ~ ., data = s1.50k.nopp.train, 
                   method = 'leapForward', 
                   tuneGrid = data.frame(nvmax = 1:7), 
                   trControl = train.control)


# Train Model 7:  Stepwise Selection Using Processed Data
train.control = trainControl(method = 'cv', number = 10)
m6.forward$results
summary(m6.forward$finalModel)     

m7.backward = train(duration ~ ., data = s1.50k.pp.train, 
                    method = 'leapBackward', 
                    tuneGrid = data.frame(nvmax = 1:7), 
                    trControl = train.control)
m7.backward$results
summary(m7.backward$finalModel)     

# How can you make a prediction suing train?


# Need to try Ridge & Lasso








