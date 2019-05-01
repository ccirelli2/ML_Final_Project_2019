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

## DROP SPEED_____________________________________________________________________________
s1.50k.nolimits$speed  <- NULL                          
s2.100k.nolimits$speed <- NULL
s3.250k.nolimits$speed <- NULL
s4.50k.wlimits$speed   <- NULL
s5.100k.wlimits$speed  <- NULL
s6.250k.wlimits$speed  <- NULL

# RANDOMIZE DATA__________________________________________________________________________
s1.50k.nolimits_ran    = s1.50k.nolimits[sample(nrow(s1.50k.nolimits)),]
s2.100k.nolimits_ran   = s2.100k.nolimits[sample(nrow(s2.100k.nolimits)),]
s3.250k.nolimits_ran   = s3.250k.nolimits[sample(nrow(s3.250k.nolimits)),]
s4.50k.wlimits_ran     = s4.50k.wlimits[sample(nrow(s4.50k.wlimits)), ]
s5.100k.wlimits_ran    = s5.100k.wlimits[sample(nrow(s5.100k.wlimits)), ]
s6.250k.wlimits_ran    = s6.250k.wlimits[sample(nrow(s6.250k.wlimits)), ]



# TRAIN / TEST SPLIT______________________________________________________________________

# Calculate Number of Training Observations
train_nrows_50k  = (nrow(s1.50k.nolimits)  * .7)
train_nrows_100k = (nrow(s2.100k.nolimits)   * .7)
train_nrows_250k = (nrow(s3.250k.nolimits)   * .7)

# Train
s1.train = s1.50k.nolimits_ran[1:   train_nrows_50k, ]
s2.train = s2.100k.nolimits_ran[1:  train_nrows_100k, ]
s3.train = s3.250k.nolimits_ran[1:  train_nrows_250k, ]
s4.train = s4.50k.wlimits_ran[1:    train_nrows_50k, ]
s5.train = s5.100k.wlimits_ran[1:   train_nrows_100k, ]
s6.train = s6.250k.wlimits_ran[1:   train_nrows_250k, ]

# Test
s1.test = s1.50k.nolimits_ran[train_nrows_50k:    nrow(s1.50k.nolimits_ran), ] # Index from training to total
s2.test = s2.100k.nolimits_ran[train_nrows_100k:  nrow(s2.100k.nolimits_ran), ]
s3.test = s3.250k.nolimits_ran[train_nrows_250k:  nrow(s3.250k.nolimits_ran), ]
s4.test = s4.50k.wlimits_ran[train_nrows_50k:     nrow(s4.50k.wlimits_ran), ]
s5.test = s5.100k.wlimits_ran[train_nrows_100k:   nrow(s5.100k.wlimits_ran), ]
s6.test = s6.250k.wlimits_ran[train_nrows_250k:   nrow(s6.250k.wlimits_ran), ]


# RF  TUNING PARAMETERS________________________________________________________________________
'replace:   Sampling with replacement, set to True
 cutoff:    
 strata:    *A factor variable that is used for stratified sampling.  How can we turn this parameter on?
 sampsize:  nrow(x), size of samples to draw. 
 nodesize:  minimum size of terminal nodes.  *setting this number larger causes smaller trees.  default for regression = 5
 maxnodes:  maximum number of terminal nodes trees in the forest can have.  *If not given, trees are grown to the max. 
 ntree:     number of trees to grow
 mtry:      For regression it is p / 3
 '

# RANGER ______________________________________________________________________________________
'ranger(formula = NULL, data = NULL, num.trees = 500, mtry = NULL,
  importance = "none", write.forest = TRUE, probability = FALSE,
  min.node.size = NULL, replace = TRUE, sample.fraction = ifelse(replace,
  1, 0.632), case.weights = NULL, class.weights = NULL, splitrule = NULL,
  num.random.splits = 1, alpha = 0.5, minprop = 0.1,
  split.select.weights = NULL, always.split.variables = NULL,
  respect.unordered.factors = NULL, scale.permutation.importance = FALSE,
  keep.inbag = FALSE, holdout = FALSE, quantreg = FALSE,
  num.threads = NULL, save.memory = FALSE, verbose = TRUE, seed = NULL,
  dependent.variable.name = NULL, status.variable.name = NULL,
  classification = NULL)
'

# TRAIN()______________________________________________________________________________________
'Fit Predictive Models over Different Tuning Parameters

Description

This function sets up a grid of tuning parameters for a number of classification and regression routines, fits each model and calculates a resampling based performance measure.

Usage

train(x, ...)

## Default S3 method:
train(x, y, method = "rf", preProcess = NULL, ...,
  weights = NULL, metric = ifelse(is.factor(y), "Accuracy", "RMSE"),
  maximize = ifelse(metric %in% c("RMSE", "logLoss", "MAE"), FALSE, TRUE),
  trControl = trainControl(), tuneGrid = NULL,
  tuneLength = ifelse(trControl$method == "none", 1, 3))
'


# COMPARE MODELS_________________________________________________________________________________

# Train Base Model with randomForest (Check run-time & rse)
t1     = Sys.time()
m.01     = randomForest(duration ~ ., data=s1.train, ntrees = 10, tuneLength = 5)
m.01.rse = sqrt(m.01$mse)    
m.01.rse
print(paste('Random Forest Run Time',Sys.time() - t1))



# Train Base Model with Ranger (Check run-time & rse)
t1     = Sys.time()
m.02     = ranger(duration ~ ., data=s1.train, num.trees = 50)
m.02.rse = sqrt(m.02$prediction.error)
m.02.rse
print(paste('Ranger Run Time', Sys.time() - t1))

# Results
'Run-time:     randomForest takes 23 seconds versus rangers 0.6 seconds 
 RSE:          random forest 129.7 vs ranger 126.6
'

# M1    TRAIN BASE MODEL WITH RANGER______________________________________________________________
'Notes:              Since RF includes bagging it has a natural cross validation
 write.forest        Save ranger.forest object, required for prediction. Set to FALSE to reduce memory usage if no prediction intended 
 prediction.error	   Overall out of bag prediction error. For classification this is the fraction of missclassified samples, 
                     for probability estimation the Brier score, for regression the mean squared error and for survival one 
                     minus Harrells C-index.
'
m1 = ranger(duration ~., data = s6.250k.wlimits_ran, num.trees = 50, write.forest = T)
# Out of Bag CV RSE
m1.oob.rse = sqrt(m1$prediction.error)
m1.oob.rse
# Test RSE Using Foreign Datase
m1.predict = predict(m1, s3.250k.nolimits)
m1.test.rse = sqrt(sum((s3.250k.nolimits$duration - m1.predict$predictions)^2) / (length(m1.predict$predictions)-2))
m1.test.rse
m1

# M2    HYPER PARAMETER SELECTION - NUBMER OF TREES____________________________________________________
'http://www.rpubs.com/Mentors_Ubiqum/tunegrid_tunelength
'

# Test Number of Trees
list.ntrees = c()
list.oob.rse = c()
list.test.rse = c()
Count = 1

rf_num_trees = function(data.train, data.test, list.ntrees, list.oob.rse, list.test.rse, Count, i){
  list.ntrees[Count] <<- i
  # Train Model
  print(paste('Training Model Using Ntrees => ', i))
  m0 = ranger(duration ~., data = data.train, num.trees = i)
  # Generate OOB RSE
  print('Generating OOB RSE')
  m0.oob.rse            = round(sqrt(m0$prediction.error),4)
  list.oob.rse[Count]   <<- m0.oob.rse
  print(paste('OOB RSE => ', m0.oob.rse))
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
  }

# Run over sequence
for (i in seq(100, 400, 100)){
  rf_num_trees(s6.250k.wlimits_ran, s1.50k.nolimits, list.ntrees, list.oob.rse, list.test.rse, Count, i)
}

list.ntrees = c(100,200,300,400)
#Create DataFrame
df = data.frame(row.names = list.ntrees)
df$oob.rse     = list.oob.rse
df$test.rse    = list.test.rse


# Graph Results
p = ggplot() + 
  geom_line(data = df, aes(x = list.ntrees, y = df$oob.rse, color = 'OOB RSE')) +
  geom_line(data = df, aes(x = list.ntrees, y = df$test.rse, color = 'Test RSE')) +
  xlab('Number of Trees') + 
  ylab('RSE') 

print(p+ ggtitle('RANDOM FOREST - TRAINING & TEST RSE'))




# M3    HYPER PARAMETER SELECTION - MTRY______________________________________________
'ntrees:  optimal seems to be 200
 mtry:    should be a range of 1 to p
'


# Test Number of Trees
list.nmtry = c()
list.oob.rse = c()
list.test.rse = c()
Count = 1

rf_num_trees = function(data.train, data.test, list.ntrees, list.oob.rse, list.test.rse, Count, i){
  'i = number of mtry.  Should be a range of 1-p'
  # Update 
  list.nmtry[Count]     <<- i
  # Train Model
  print(paste('Training Model Using N-MTRY => ', i))
  m0 = ranger(duration ~., data = data.train, num.trees = 400, mtry = i)
  # Generate OOB RSE
  print('Generating OOB RSE')
  m0.oob.rse            = round(sqrt(m0$prediction.error),4)
  list.oob.rse[Count]   <<- m0.oob.rse
  print(paste('OOB RSE => ', m0.oob.rse))
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
}

# Run over sequence
ncolnames = length(colnames(s1.50k.nolimits))

# Iterate over number of mtry
for (i in seq(1, ncolnames-1, 1)){
  rf_num_trees(s6.250k.wlimits_ran, s1.50k.nolimits, list.nmtry, list.oob.rse, list.test.rse, Count, i)
}

list.nmtry

#Create DataFrame
df = data.frame(row.names = list.nmtry[1:9])
df$oob.rse     = list.oob.rse
df$test.rse    = list.test.rse

# Graph Results
p = ggplot() + 
  geom_line(data = df, aes(x = list.nmtry[1:9], y = df$oob.rse, color = 'OOB RSE')) +
  geom_line(data = df, aes(x = list.nmtry[1:9], y = df$test.rse, color = 'Test RSE')) +
  xlab('Number of Features') + 
  ylab('RSE') 

print(p+ ggtitle('RANDOM FOREST - TRAINING & TEST RSE'))




# M3    HYPER PARAMETER SELECTION - ALPHA______________________________________________
'alpha	For "maxstat" splitrule: Significance threshold to allow splitting.
        Default = 0.5
'

# Test Number of Trees
list.alpha = c()
list.oob.rse = c()
list.test.rse = c()
Count = 1

rf_num_trees = function(data.train, data.test, list.ntrees, list.oob.rse, list.test.rse, Count, i){
  'i = value for alpha.'
  # Update 
  list.alpha[Count]     <<- i
  # Train Model
  print(paste('Training Model Using Alpha => ', i))
  m0 = ranger(duration ~., data = data.train, num.trees = 400, mtry = 5, alpha = i)
  # Generate OOB RSE
  print('Generating OOB RSE')
  m0.oob.rse            = round(sqrt(m0$prediction.error),4)
  list.oob.rse[Count]   <<- m0.oob.rse
  print(paste('OOB RSE => ', m0.oob.rse))
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
}

# Iterate over number of mtry
for (i in seq(0.5, 0.01, -0.05)){
  rf_num_trees(s6.250k.wlimits_ran, s1.50k.nolimits, list.nmtry, list.oob.rse, list.test.rse, Count, i)
}



#Create DataFrame
df = data.frame(row.names = list.alpha)
df$oob.rse     = list.oob.rse
df$test.rse    = list.test.rse
list.oob.rse
# Graph Results
p = ggplot() + 
 #geom_line(data = df, aes(x = list.alpha, y = df$oob.rse, color = 'OOB RSE')) +
  geom_line(data = df, aes(x = list.alpha, y = df$test.rse, color = 'Test RSE')) +
  xlab('ALPHA') + 
  ylab('RSE') 

print(p+ ggtitle('RANDOM FOREST - ALPHA - TEST RSE'))


# Try using trian and passing the cp parameter there. Or just train a different model and set the values to 200 trees w/ mty = 10




# M3    HYPER PARAMETER SELECTION - MIN NODE SZE______________________________________________
'min.node.size =       This is the minimum node size, in the example above the minimum node size is 10. This parameter 
                       implicitly sets the depth of your trees. Minimum size of terminal nodes. Setting this number 
                       larger causes smaller trees to be grown (and thus take less time). Note that the default values are different 
                       for classification (1) and regression (5).
'
# Test Number of Trees
list.node.size = c()
list.oob.rse = c()
list.test.rse = c()
Count = 1

rf_num_trees = function(data.train, data.test, list.ntrees, list.oob.rse, list.test.rse, Count, i){
  'i = value for alpha.'
  # Update 
  list.node.size[Count]     <<- i
  # Train Model
  print(paste('Training Model Using Min.Node.Size => ', i))
  m0 = ranger(duration ~., data = data.train, num.trees = 200, mtry = 5, alpha = 0.25, min.node.size = i)
  # Generate OOB RSE
  print('Generating OOB RSE')
  m0.oob.rse            = round(sqrt(m0$prediction.error),4)
  list.oob.rse[Count]   <<- m0.oob.rse
  print(paste('OOB RSE => ', m0.oob.rse))
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
}

# Iterate over number of nodes
for (i in seq(2, 10, 2)){
  rf_num_trees(s6.250k.wlimits_ran, s1.50k.nolimits, list.nmtry, list.oob.rse, list.test.rse, Count, i)
}


#Create DataFrame
df = data.frame(row.names = list.node.size)
df$oob.rse     = list.oob.rse
df$test.rse    = list.test.rse
list.oob.rse

# Graph Results
p = ggplot() + 
#  geom_line(data = df, aes(x = list.node.size, y = df$oob.rse, color = 'OOB RSE')) +
  geom_line(data = df, aes(x = list.node.size, y = df$test.rse, color = 'Test RSE')) +
  xlab('MIN.NODE.SIZE') + 
  ylab('RSE') 

print(p+ ggtitle('RANDOM FOREST - MIN.NODE.SIZE - TEST RSE'))






