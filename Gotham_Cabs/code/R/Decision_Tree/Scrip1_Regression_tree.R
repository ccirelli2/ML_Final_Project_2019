# READINGS_______________________________________________________________________________
'Train Regression Tree using rpart:
 - https://www.statmethods.net/advstats/cart.html

Regression Trees using Tree
 - http://www.di.fc.ul.pt/~jpn/r/tree/tree.html
'

## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(rpart)
library(tree)

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



# TRAIN REGRESSION TREE - TREE___________________________________________________

# Train Model 1 -----------------------------------------------------------------
m1.train = tree(duration ~ ., data = s6.train)
# Plot Tree
plot(m1.train)
text(m1.train, cex = .75)                   # Note how all fo the first splits are distance or speed. 
# Get Summary Statistics
m1.summary = summary(m1.train)
m1.summary                                  # Note 15 nodes
# Calculate RSE
m1.train.rse = sqrt(sum((m1.summary$residuals^2)) / (length(m1.summary$residuals) -2) )
print(paste('Model-1 train rse =>', m1.train.rse))
# Make a Prediction
m1.predict = predict(m1.train, s6.test)
m1.test.rse = sqrt(sum((s6.test$duration - m1.predict)^2) / (length(s6.test$duration) -2) )
print(paste('Model-1 test rse =>', m1.test.rse))



# Train Model 2 -----------------------------------------------------------------
m2.train = tree(duration ~ ., data = s6.train, mindev = 0.001)           # Controls the number of nodes.  Default = 0.01
# Plot Tree
plot(m2.train)
text(m2.train, cex = .75)                   # Note how all fo the first splits are distance or speed. 
# Get Summary Statistics
m2.summary = summary(m2.train)
m2.summary                                  # Note 15 nodes
# Calculate RSE
m2.train.rse = sqrt(sum((m2.summary$residuals^2)) / (length(m2.summary$residuals) -2) )
print(paste('Model-1 train rse =>', m2.train.rse))
# Make a Prediction
m2.predict = predict(m2.train, s6.test)
m2.test.rse = sqrt(sum((s6.test$duration - m2.predict)^2) / (length(s6.test$duration) -2) )
print(paste('Model-1 test rse =>', m2.test.rse))


# Create Lists to Capture Values & A DataFrame to House the Columns
index.mindev = c()
list.train.rse = c()
list.test.rse  = c()
list.test.unknowndata.rse = c()
index.count = 1

# Iterate Over Range of Values for mindev
for (i in seq(0.0001, 0.01, 0.00025)){
  # Train Model
  index.mindev[index.count] = i
  m0.train = tree(duration ~ ., data = s6.train, mindev = i)           # Controls the number of nodes.  Default = 0.01
  # Plot Tree
  #plot(m0.train)
  #text(m0.train, cex = .75)                   # Note how all fo the first splits are distance or speed. 
  # Get Summary Statistics
  m0.summary = summary(m0.train)
  m0.summary                                  # Note 15 nodes
  # Calculate RSE
  m0.train.rse = sqrt(sum((m0.summary$residuals^2)) / (length(m0.summary$residuals) -2) )
  list.train.rse[index.count] = m0.train.rse
  print(paste('Model-1 ', 'mindev => ', i, 'train rse =>', m0.train.rse))
  # Make a Prediction
  m0.predict = predict(m0.train, s6.test)
  m0.test.rse = sqrt(sum((s6.test$duration - m0.predict)^2) / (length(s6.test$duration) -2) )
  list.test.rse[index.count] = m0.test.rse
  print(paste('Model-1 ', 'mindev => ', i , 'test rse =>', m0.test.rse))
  # Make Prediction - Unseen Dataset
  m.unknown.predict = predict(m0.train, s4.50k.wlimits_ran)
  m.unknowndata.test.rse = sqrt(sum((s4.50k.wlimits_ran$duration - m.unknown.predict)^2) / (length(s6.test$duration) -2) )
  list.test.unknowndata.rse[index.count] = m.unknowndata.test.rse
  print(paste('Model-1 ', 'mindev => ', i , 'unknown data test rse =>', m.unknowndata.test.rse))
  print('-------------------------------------------------------------')
  index.count = index.count + 1
}

# Create DataFrame
df.0 = data.frame(row.names = index.mindev)
df.0$train.rse = list.train.rse
df.0$test.rse = list.test.rse
df.0$unknowndata.test.rse = list.test.unknowndata.rse
df.0


