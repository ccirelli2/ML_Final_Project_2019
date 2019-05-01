# GENERATE FINAL PREDICTION

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


# IMPORT THE TEST SET_______________________________________________________________________________
test.file.location = '/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/test_submission/'
test.file.name = 'Final_test_data_clean_version.csv'

s.final.test = read.csv('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/test_submission/Final_test_data_clean_version.csv')
s.final.test = s.final.test[2:10]
head(s.final.test)
head(s1.test)


# M3    HYPER PARAMETER SELECTION - MIN NODE SZE______________________________________________
'min.node.size =       This is the minimum node size, in the example above the minimum node size is 10. This parameter 
implicitly sets the depth of your trees. Minimum size of terminal nodes. Setting this number 
larger causes smaller trees to be grown (and thus take less time). Note that the default values are different 
for classification (1) and regression (5).
'


rf_num_trees = function(data.train, data.test){
  'i = value for alpha.'
  # Update 
  list.node.size[Count]     <<- i
  # Train Model
  print(paste('Training Model Using Min.Node.Size => ', i))
  m0 = ranger(duration ~., data = data.train, num.trees = 200, mtry = 5, alpha = 0.25, min.node.size = 1)

    # Generate OOB RSE
  print('Generating OOB RSE')
  m0.oob.rse            = round(sqrt(m0$prediction.error),4)
  list.oob.rse[Count]   <<- m0.oob.rse
  print(paste('OOB RSE => ', m0.oob.rse))
  
  # Generate Prediction Using New Sample Data
  print('Generating Test Prediction')
  m0.predict            = predict(m0, data.test)
  # Calculate Test RSE
  
  print('Model Completed.  Returning model object to user')
  print('-----------------------------------------------------------------------------')
}

rf_num_trees(s6.250k.wlimits_ran, s.final.test, list.nmtry, list.oob.rse, list.test.rse, Count, i)



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