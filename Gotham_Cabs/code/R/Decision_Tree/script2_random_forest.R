## READINGS_______________________________________________________________________________
'Random Forest Explained
 - https://uc-r.github.io/random_forests
    ntree: number of trees. We want enough trees to stabalize the error but using too many trees is unncessarily inefficient, especially when using large data sets.
    mtry: the number of variables to randomly sample as candidates at each split. When mtry =p the model equates to bagging. When mtry =1
          the split variable is completely random, so all variables get a chance but can lead to overly biased results. A common suggestion is to start with 5 values evenly spaced across the range from 2 to p.
    sampsize: the number of samples to train on. The default value is 63.25% of the training set since this is the expected value of unique observations in the bootstrap sample. Lower sample sizes can reduce the training time but may introduce more bias than necessary. Increasing the sample size can increase performance but at the risk of overfitting because it introduces more variance. Typically, when tuning this parameter we stay near the 60-80% range.
    nodesize: minimum number of samples within the terminal nodes. Controls the complexity of the trees. Smaller node size allows for deeper, more complex trees and smaller node results in shallower trees. This is another bias-variance tradeoff where deeper trees introduce more variance (risk of overfitting) and shallower trees introduce more bias (risk of not fully capturing unique patters and relatonships in the data).
    maxnodes: maximum number of terminal nodes. Another way to control the complexity of the trees. More nodes equates to deeper, more complex trees and less nodes result in shallower trees.
'

## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(ggplot2)
library(randomForest)
library(ranger)               # Faster implementation of Random Forest
library(tree)
library(ISLR)
library(MASS)

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


# M1 - RANDOM FOREST MODEL____________________________________________________________
t1 = Sys.time()
m1 = randomForest(duration ~ ., data = s6.train, ntree = 50, mtry = 11)
print(Sys.time() - t1)

# Get Model Data
m1
# Plot Model
'Will show you how the MSE falls versus the number of trees generated'
plot(m1, main = 'Random Forest Tree - Plot Error vs Num Trees')                              # Looks like its bottoming out around 35
# Number of trees with lowest MSE (Elbow point)
min.rse = which.min(m1$mse)
min.rse
# Feature Importance
importance(m1)
varImpPlot(m1, main = 'Variable Importance Plot - Node Purity')
# Get Tree With Best RSE
m1.min.rse    = sqrt(m1$mse[which.min(m1$mse)])
m1.min.rse

# M2 -  - TRAINING / TEST SPLIT_________________________________________
t1 = Sys.time()
m2 = randomForest(duration ~ ., data = s1.50k.nolimits_ran, xtest = s1.test[,2:11], ytest = s1.test$duration, ntree = 50)
print(Sys.time() - t1)

# Extract OOB and CV ERRORS
oob_error = sqrt(m2$mse)
cv_error  = sqrt(m2$test$mse)
m2.index = seq(1: m2$ntree)

# Generate Plot
p = ggplot() + 
  geom_line(aes(x = m2.index, y = oob_error, color = 'TEST RSE')) +
  geom_line(aes(x = m2.index, y = cv_error, color = 'OOB RSE')) +
  xlab('Number of Trees') + 
  ylab('RSE') 
print(p+ ggtitle('Compare Out of Bag & Test (CV) RSE'))

# Get Min RSE
which.min(m2$mse)

# RMSE Of Optimal Random Forest
sqrt(m2$mse[which.min(m2$mse)])


# M3 - RANGER - FASTER IMPLEMENTATION RANDOM FOREST_________________________________________

m3 <- ranger(formula   = duration ~ ., 
             data      = s1.train, 
             num.trees = 50,
             mtry      = 3)
m3.predict = predict(m3, s1.test)
m3.rse     = sqrt(sum((s1.test$duration - m3.predict$predictions)^2) / (length(s1.test$duration) -2))


# M4    INITIAL TUNING (MTRY)______________________________________________________________
'Optomize based on different values for mtry

randomForest::tuneRF() ** Not working. 
'
# Create Lists to Capture Model Output
list.m0.mtry      = c()
list.m0.train.rse = c()
list.m0.test.rse  = c()

# Iterate Over Different Values for MTRY
for (i in seq(1,10)){
  # Train Model
  print(paste('Training model for MTRY =>', i))
  m0 =   ranger(formula   = duration ~ ., 
                data      = s4.train, 
                num.trees = 500,
                mtry      = i)
  # Generate Train Error
  list.m0.train.rse[i] = sqrt(m0$prediction.error)
  print('Generating Prediction')
  m0.predict = predict(m0, s4.test)
  m0.test.rse     = sqrt(sum((s4.test$duration - m0.predict$predictions)^2) / (length(s4.test$duration) -2))
  list.m0.mtry[i] = i
  list.m0.test.rse[i]  = round(m0.test.rse,4)
  print(paste('Model => ', i, ' RSE =>', round(m0.rse,4)))
  print('--------------------------------------------------')
}

# Create Data frame to house results
df = data.frame(row.names = list.m0.mtry)
df$RSE.Train = list.m0.train.rse
df$RSE.Test  = list.m0.test.rse

# Graph Results
p = ggplot() + 
  geom_line(data = df, aes(x = list.m0.mtry, y = list.m0.train.rse, color = 'Train RSE')) +
  geom_line(data = df, aes(x = list.m0.mtry, y = list.m0.test.rse, color = 'Test RSE')) +
  xlab('MTRY Value') + 
  ylab('RSE') 

print(p+ ggtitle('Random Forest - MTRY Values - RSE'))


# M5 - FULL GRID SEARCH____________________________________________________________________

# Create Hypergrid of features (dataframe of all combinations of the supplied vectors)
hyper_grid = expand.grid(
             mtry        = seq(2,10, by = 2), 
             node_size   = seq(2,10, by = 2), 
             #sample_size = c(.7),
             OOB_RMSE    = 0)

hyper_grid
# Check Number of Model Variations
nrow(hyper_grid)

# Iterate hypergrid while training model
for(i in 1:nrow(hyper_grid)){
  # train model
  model <- ranger(
    formula         = duration ~ ., 
    data            = s1.50k.nolimits_ran, 
    num.trees       = 50,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    #sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )}


# Add OOB Error to Grid
hyper_grid$OOB_RMSE[i] = sqrt(model$prediction.error)

hyper_grid





