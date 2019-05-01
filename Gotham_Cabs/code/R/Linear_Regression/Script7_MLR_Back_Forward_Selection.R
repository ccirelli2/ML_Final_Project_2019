## READING MATERIALS______________________________________________________________________
'
How to Perform Stepwise Selection:
- http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/154-stepwise-regression-essentials-in-r/
Definitions:  RMSE, RSE, MAE, RAE:
- https://www.saedsayad.com/model_evaluation_r.htm
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

## DROP SPEED_____________________________________________________________________________
s1.50k.nolimits$speed  <- NULL                          
s2.100k.nolimits$speed <- NULL
s3.250k.nolimits$speed <- NULL
s4.50k.wlimits$speed   <- NULL
s5.100k.wlimits$speed  <- NULL
s6.250k.wlimits$speed  <- NULL

# SET SEED FOR ENTIRE CODE________________________________________________________________
set.seed(123)                                                                 

# RANDOMIZE DATA__________________________________________________________________________
s1.50k.nolimits_ran    = s1.50k.nolimits[sample(nrow(s1.50k.nolimits)),]
s2.100k.nolimits_ran   = s2.100k.nolimits[sample(nrow(s2.100k.nolimits)),]
s3.250k.nolimits_ran   = s3.250k.nolimits[sample(nrow(s3.250k.nolimits)),]
s4.50k.wlimits_ran     = s4.50k.wlimits[sample(nrow(s4.50k.wlimits)), ]
s5.100k.wlimits_ran    = s5.100k.wlimits[sample(nrow(s5.100k.wlimits)), ]
s6.250k.wlimits_ran    = s6.250k.wlimits[sample(nrow(s6.250k.wlimits)), ]


# TRAIN MODELS____________________________________________________________________________

# Define Train Control Object
train.control = trainControl(method = 'cv', number = 10)

# FORWARD SELECTION ----------------------------------------------------------------------
m6.forward = train(duration ~ ., data = s6.250k.wlimits_ran, 
                   method = 'leapBackward', 
                   tuneGrid = data.frame(nvmax = 2:9), 
                   trControl = train.control)
print('hello world')
# Get Results
m6.summary = summary(m6.forward$finalModel)     
m6.forward$bestTune
sqrt(m6.summary$rss/length(s6.250k.wlimits_ran$duration))

# Plot Results RMSE vs Number of Features
plot(m6.forward$results$RMSE, main = 'MLR - Forward Selection - RMSE', xlab = 'Number of Features')


# BACKWARD SELECTION----------------------------------------------------------------------

m7.backward = train(duration ~ ., data = s6.250k.wlimits_ran, 
                    method = 'leapForward', 
                    tuneGrid = data.frame(nvmax = 1:9), 
                    trControl = train.control)
m7.backward
m7.backward$results
summary(m7.backward$finalModel)     

print('hello world')










# M1:     BACKWARD SELECTION_____________________________________________________________

# Create Training Method - CV & 10kfold
train.control = trainControl(method = 'cv', number = 10)

model_opt <- function(dataset, opt_method, num_param, cv_grid, result2return){
  'opt_method:        either leapBackward or leapForward
  results:           A dataframe with the training error rate and values for the tuning params
  bestTune           A dataframe with the final parameters
  finalModel         Aa fit object using the best parameters'
  # Train Model
  m0 = train(duration ~ dataset$pickup_x, dataset$pickup_y, dataset$dropoff_x, dataset$dropoff_y, 
             dataset$weekday, dataset$hour_, dataset$day_, dataset$distance, dataset$month_, 
             data      = dataset, 
             method    = opt_method,                                   # Step selection
             tuneGrid  = data.frame(nvmax = 1 : num_param),            # Number of features to consider in the model
             trControl = cv_grid)                                      # Cross Validation technique 
  # Results
  if (result2return == 'results'){
    print(paste('Results:',m0$results))
    return(m0$results)}
  
  # Final Model - # An asterisk specifies that the feature was included in the model
  else if (result2return == 'finalModel'){
    print(summary(m0$finalModel))
    return(summary(m1.backward$finalModel))}      
  
  else if (result2return == 'bestTune'){
    print(paste('Best Tune:', m0$bestTune))
    return(m0$bestTune)}
}


m0.output = model_opt(s4.50k.wlimits_ran, 'leapForward', 9, train.control, 'bestTune')


param_index = seq(from = 1, to = 9, by = 1)
m0.output$RMSE

plot(m0.output$RMSE, type = 'o', names.arg = param_index, main = 'FORKWARD SELECTION - RMSE FOR NUM OF PARAMETERS', 
     xlab = 'Number of Parameters', ylab = 'RMSE')






