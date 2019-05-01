## REFERENCES_____________________________________________________________________________
'MLR Output:      https://www.graphpad.com/support/faq/standard-deviation-of-the-residuals-syx-rmse-rsdr/
                  http://www.sthda.com/english/articles/38-regression-model-validation/158-regression-model-accuracy-metrics-r-square-aic-bic-cp-and-more/
'

## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(lattice)
library(ggplot2)
library(caret)  # used for parameter tuning

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



## M1:    MULTILINEAR REGRESSION - WITH POLYNOMIALS______________________________________________________

mlr.poly <- function(data_train, data_test, polynomial, objective) {
  # Train Model  
  m1.mlr.poly = lm(duration ~ poly(pickup_x, pickup_y, dropoff_x, dropoff_y, 
                                   weekday, hour_,  day_,distance, 
                                   month_,
                                   degree = polynomial, raw = T), data = data_train)
  # Return Summary Stats
  if (objective == 'train'){
    m1.summary               = summary(m1.mlr.poly)
    train.rse                = sqrt((sum(m1.summary$residuals^2)) / length(m1.summary$residuals) -2)
    return(train.rse)
  }
    # Generate Prediction
    if (objective == 'test'){ 
      m1.mlr.poly.predict    = predict(m1.mlr.poly, data_test)
      test.rse               = sqrt(sum((data_test$duration - m1.mlr.poly.predict)^2) / (length(m1.mlr.poly.predict) -2))
      return(test.rse)
    }
}


# Create Lists to Capture Values & A DataFrame to House the Columns
index.rse = c()
list.train.rse = c()
list.test.rse  = c()
index_count = 1

# Iterate Over a Sequence of Values For The Polynomials
for (i in seq(1, 3.5, 0.1)){
  index.rse[index_count] = i
  print(paste('creating test set for poly =>', i))
  train.rse = mlr.poly(s6.train, s6.test, i, 'train')
  list.train.rse[index_count] = round(train.rse, 2)  
  print(paste('creating train set for poly =>', i))
  test.rse = mlr.poly(s1.train, s1.test, i, 'test')
  print(paste('Test RSE', round(test.rse,2)))
  list.test.rse[index_count] = round(test.rse,2)
  index_count = index_count + 1
}


# Create DataFrame
df = data.frame(row.names = index.rse)
df$index.rse = index.rse
df$train.rse = list.train.rse
df$test.rse = list.test.rse
df

# Generate a Plot for Train & Test Points
p = ggplot() + 
  geom_line(data = df, aes(x = df$index.rse, y = df$train.rse, color = 'Train RSE')) +
  geom_line(data = df, aes(x = df$index.rse, y = df$test.rse, color = 'Test RSE')) +
  xlab('Number of Polynomials') + 
  ylab('RSE') 

print(p+ ggtitle('Plot Test & Training Polynomials'))





