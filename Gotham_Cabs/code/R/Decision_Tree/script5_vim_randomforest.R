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
  m0 = ranger(duration ~., data = data.train, num.trees = 200, mtry = 10, alpha = 0.1, min.node.size = i)
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
for (i in seq(1, 10, 1)){
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






