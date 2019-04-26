E# DOCUMENTATION__________________________________________________________________________
'Usage
rpart(formula, data, weights, subset, na.action = na.rpart, method,
      model = FALSE, x = FALSE, y = TRUE, parms, control, cost, ...)

 Vid:  Tuning Hyperparameters
       https://www.youtube.com/watch?v=1rNclbWruI0

 Understaning output from printcp():
      https://stackoverflow.com/questions/9666212/how-to-compute-error-rate-from-a-decision-tree
      https://stackoverflow.com/questions/29197213/what-is-the-difference-between-rel-error-and-x-error-in-a-rpart-decision-tree

Explanation Output
    https://community.alteryx.com/t5/Alteryx-Knowledge-Base/Understanding-the-Outputs-of-the-Decision-Tree-Tool/ta-p/144773
'

## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(rpart)
library(rpart.plot)
library(tree)
library(ggplot2)

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



# MODEL1___________________________________________________________________________________

# Model1 - Not tuning hyperparameters
'method:  for regression we use "anova"
 m1.train.output:
          - n         number of observations
          - split     the feature on which the tree split
          - n         how many observations 
          - deviance  missclassification.  should be equivalent to MSE. 
          - yval      prediction
          - *         denotes a terminal node
'
m1.train = rpart(duration ~ ., data = s6.train, method = 'anova')

# Model1 - Plot
'Interpretation:
          - 1st val at terminal node:   Prediction. 
          - 2nd val                     n = number of observations that ended up in each leaf. 
          - 3rd val                     % of observations that ended up in each leaf. 
'
rpart.plot(m1.train, type = 3, extra = 101, fallen.leaves = T, main = 'Regression Tree - M1')

# Model1 - Calculate Train RSE
m1.residuals = residuals(m1.train)
m1.train.rse = sqrt(sum(m1.residuals^2) / (length(m1.residuals) - 2))
print(m1.train.rse)

# Model1 - Generate a Prediction
m1.predict   = predict(m1.train, s1.test)

# Model1 - Calculate Test RSE
m1.test.rse  = sqrt(sum((s1.test$duration - m1.predict)^2) / (length(s1.test$duration) - 2)) 
m1.test.rse



# MODEL2_____________________________________________________________________________________
'parms:     determine what tree you want.  options 1.) information, 2.) gini, 3.) anova.  
            only anova is permissible for regression. 
 cp:        - complexity pattern.  trees are prone to overfitting. 
              used to give some constraint to that overfitting. 
            - it is a number or amount by which the splitting the tree would
              degrees the relative error.  .2 is very strict.  any variable 
              that will reduce the error by .2 will be split.  if lower than 
            - .2 it will not give you the split. 
 minsplit	  the minimum number of observations that must exist in a node in order for a split to be attempted.
 minbucket	the minimum number of observations in any terminal <leaf> node. If only one of minbucket or minsplit is specified, the code 
            either sets minsplit to minbucket*3 or minbucket to minsplit/3, as appropriate.
 xval       Number of cross validations
'
# Train Model
m2.train = rpart(duration ~ ., data = s6.train, method = 'anova', 
                 control = rpart.control(cp = .0016, minsplit = 5, minbucket = 5, maxdepth = 10))

# Plot Tree
rpart.plot(m2.train, type = 5, extra = 101, fallen.leaves = T, main = 'Regression Tree - M2')

# Calculate Test Residual
m2.residuals = residuals(m2.train)
m2.train.rse = sqrt(sum(m2.residuals^2) / (length(m2.residuals) - 2))
print(paste('Train RSE =>', round(m2.train.rse, 4)))

# Make Predictin - Unpurned Tree
m2.unpruned.predict = predict(m2.train, s6.test)
m2.unpruned.test.rse = sqrt(sum((s6.test$duration - m2.unpruned.predict)^2) / (length(s6.test$duration) -2) )
print(paste('Model-2 Test RSE =>', round(m2.unpruned.test.rse, 4)))                  # Pre-pruning test erorr was better. 

# Method For Tuning 
'- Method
                  Grow Tree to Full Size
                  Prune it back
 - Xval:          default parameter, set at 10.  Sounds like Kfold. Used for cross validation. 
                  It gives you the error for each fit. 
 - Plot:          Top axis = size of the tree, Bottom axis = cp value, Xval = error?
                  *You can use the top axis to figure out where you should stop growing your tree.  here it looks like it should be 13. 
 - xerror         according to the video there should be an elbow point with the xerror
 - relative error:  (y - yi) / y 
 
 How to chose the appropriate cp?
 - step1    find the lowest cp that results from printcp
 - step2    add to it its standard deviation
 - step3    take the highest xerror that is less than this sum. 
 - step4    use this cp to prune your tree. 
'
# Print Results of Cross Validation
printcp(m2.train)
# Plot Cp - Will Plot Number of Splits vs Cp vs Error
plotcp(m2.train)

# Prune Tree
m2.prune = prune(m2.train, cp = 0.001)
rpart.plot(m2.prune, cex = .5, extra = 1)                               # Can we get auto feedback on the size of tree?

# Calculate Test RSE (Compare Pruned Tree to Original Tree)

# Make a Prediction
m2.prune.predict = predict(m2.prune, s6.test)
m2.prune.test.rse = sqrt(sum((s6.test$duration - m2.prune.predict)^2) / (length(s6.test$duration) -2) )
print(paste('Model-2 test rse =>', m2.prune.test.rse))                  # Pre-pruning test erorr was better. 


# Try a holdout dataset to guage the error rate. 
rpart.plot(m2.prune, type = 1, extra = 101, fallen.leaves = F)














