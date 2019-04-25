E# DOCUMENTATION__________________________________________________________________________
'Usage
rpart(formula, data, weights, subset, na.action = na.rpart, method,
      model = FALSE, x = FALSE, y = TRUE, parms, control, cost, ...)

 Vid:  Tuning Hyperparameters
       https://www.youtube.com/watch?v=1rNclbWruI0
'

## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(rpart)
library(rpart.plot)
library(tree)
library(ggplot2)

# CREATE DATASET_________________________________________________________________________
setwd('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/data')
s1.50k.nolimits        = read.csv('sample1_50k.csv')[2:12]                          #[2:12] drop datetime col. 
s4.50k.wlimits         = read.csv('sample2_wlimits_50k.csv')[2:12]

# RANDOMIZE DATA__________________________________________________________________________
s1.50k.nolimits_ran    = s1.50k.nolimits[sample(nrow(s1.50k.nolimits)),]
s4.50k.wlimits_ran     = s4.50k.wlimits[sample(nrow(s4.50k.wlimits)), ]

# TRAIN / TEST SPLIT______________________________________________________________________

# Calculate Number of Training Observations
train_nrows_50k  = (nrow(s1.50k.nolimits)  * .7)

# Train
s1.train = s1.50k.nolimits_ran[1:   train_nrows_50k, ]
s4.train = s4.50k.wlimits_ran[1:    train_nrows_50k, ]

# Test
s1.test = s1.50k.nolimits_ran[train_nrows_50k:    nrow(s1.50k.nolimits_ran), ] # Index from training to total
s4.test = s4.50k.wlimits_ran[train_nrows_50k:     nrow(s4.50k.wlimits_ran), ]


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
m1.train = rpart(duration ~ ., data = s1.train, method = 'anova')


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

# Model1 - Generate a Prediction
m1.predict   = predict(m1.train, s1.test)

# Model1 - Calculate Test RSE
m1.test.rse  = sqrt(sum((s1.test$duration - m1.predict)^2) / (length(s1.test$duration) - 2)) 
m1.test.rse



# MODEL2_____________________________________________________________________________________
'parms:     determine what tree you want.  options 1.) information, 2.) gini, 3.) anova.  
            only anova is permissible for regression. 
 cp:        complexity pattern.  trees are prone to overfitting. 
            used to give some constraint to that overfitting. 
            it is a number or amount by which the splitting the tree would
            degrees the relative error.  .2 is very strict.  any variable 
            that will reduce the error by .2 will be split.  if lower than 
            .2 it will not give you the split. 
 minsplit:  any node 5 number of observations will not be further split. 
            so it is the
'
m2.train = rpart(duration ~ ., data = s1.train, parms = list(split = 'anova'), 
                 control = rpart.control(cp = .1, minsplit = 1, minbucket = 1, maxdepth = 30))

m2.residuals = residuals(m2.train)
m2.train.rse = sqrt(sum(m2.residuals^2) / (length(m2.residuals) - 2))
m2.train.rse
m2.train








