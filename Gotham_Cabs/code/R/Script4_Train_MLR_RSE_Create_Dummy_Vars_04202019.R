## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(lattice)
library(ggplot2)
library(caret)  # used for parameter tuning
library(fastDummies)


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

# CREATE DUMMY VARIABLES__________________________________________________________________

## S1 - Factor
s1.50k.nolimits$pickup_x    <- factor(s1.50k.nolimits$pickup_x)
s1.50k.nolimits$pickup_y    <- factor(s1.50k.nolimits$pickup_y)
s1.50k.nolimits$dropoff_x   <- factor(s1.50k.nolimits$dropoff_x)
s1.50k.nolimits$dropoff_y   <- factor(s1.50k.nolimits$dropoff_y)
s1.50k.nolimits$weekday     <- factor(s1.50k.nolimits$weekday)
s1.50k.nolimits$hour_       <- factor(s1.50k.nolimits$hour_)
s1.50k.nolimits$day_        <- factor(s1.50k.nolimits$day_)
s1.50k.nolimits$month_      <- factor(s1.50k.nolimits$month_)
s1.50k.nolimits             <- dummy_cols(s1.50k.nolimits)              # Create Dummy Columns

s2.100k.nolimits$pickup_x    <- factor(s2.100k.nolimits$pickup_x)
s2.100k.nolimits$pickup_y    <- factor(s2.100k.nolimits$pickup_y)
s2.100k.nolimits$dropoff_x   <- factor(s2.100k.nolimits$dropoff_x)
s2.100k.nolimits$dropoff_y   <- factor(s2.100k.nolimits$dropoff_y)
s2.100k.nolimits$weekday     <- factor(s2.100k.nolimits$weekday)
s2.100k.nolimits$hour_       <- factor(s2.100k.nolimits$hour_)
s2.100k.nolimits$day_        <- factor(s2.100k.nolimits$day_)
s2.100k.nolimits$month_      <- factor(s2.100k.nolimits$month_)
s2.100k.nolimits             <- dummy_cols(s2.100k.nolimits)              # Create Dummy Columns

s4.50k.wlimits$pickup_x     <- factor(s4.50k.wlimits$pickup_x)
s4.50k.wlimits$pickup_y     <- factor(s4.50k.wlimits$pickup_y)
s4.50k.wlimits$dropoff_x    <- factor(s4.50k.wlimits$dropoff_x)
s4.50k.wlimits$dropoff_y    <- factor(s4.50k.wlimits$dropoff_y)
s4.50k.wlimits$weekday      <- factor(s4.50k.wlimits$weekday)
s4.50k.wlimits$hour_        <- factor(s4.50k.wlimits$hour_)
s4.50k.wlimits$day_         <- factor(s4.50k.wlimits$day_)
s4.50k.wlimits$month_       <- factor(s4.50k.wlimits$month_)
s4.50k.wlimits              <- dummy_cols(s4.50k.wlimits)               # Create Dummy Columns

s5.100k.wlimits$pickup_x    <- factor(s5.100k.wlimits$pickup_x)
s5.100k.wlimits$pickup_y    <- factor(s5.100k.wlimits$pickup_y)
s5.100k.wlimits$dropoff_x   <- factor(s5.100k.wlimits$dropoff_x)
s5.100k.wlimits$dropoff_y   <- factor(s5.100k.wlimits$dropoff_y)
s5.100k.wlimits$weekday     <- factor(s5.100k.wlimits$weekday)
s5.100k.wlimits$hour_       <- factor(s5.100k.wlimits$hour_)
s5.100k.wlimits$day_        <- factor(s5.100k.wlimits$day_)
s5.100k.wlimits$month_      <- factor(s5.100k.wlimits$month_)
s5.100k.wlimits             <- dummy_cols(s5.100k.wlimits)              # Create Dummy Columns



# TRAIN / TEST SPLIT______________________________________________________________________

# Train
s1.train = s1.50k.nolimits[1:  (nrow(s1.50k.nolimits)  * .7), ]
s2.train = s2.100k.nolimits[1:  (nrow(s2.100k.nolimits)  * .7), ]
s4.train = s4.50k.wlimits[1:   (nrow(s4.50k.wlimits)  * .7), ]
s5.train = s5.100k.wlimits[1:   (nrow(s5.100k.wlimits)  * .7), ]


# Test
s1.test = s1.50k.nolimits[1:  (nrow(s1.50k.nolimits)  * .3), ]
s4.test = s4.50k.wlimits[1:   (nrow(s4.50k.wlimits)  * .3), ]


## M1:    MULTILINEAR REGRESSION_______________________________________________________________________

# Train Model
m3.mlr = lm(duration ~ ., data = s1.train)
m3.summary = summary(m3.mlr)                                            # return summary
m3.summary
print('hello world')

# Make Prediction
m3.mlr.predict    = predict(m3.mlr, s1.50k.nolimits_ran)
m3.mlr.rse         = sqrt(sum((s1.50k.nolimits_ran$duration - m3.mlr.predict)^2) / (nrow(s1.50k.nolimits_ran) -2) )
m3.mlr.rse         # Residual Standard Error = 285.62




## M2:    MLR - TEST ALL DATASETS______________________________________________________________________

# Train Model - s1 & s4
training_sets <- list(s1.50k.nolimits, s2.100k.nolimits, s4.50k.wlimits, s5.100k.wlimits)
m2.results    <- data.frame('Index' = 1)
Count          = 0

# Get R2:  All Datasets---------------------------------------------------------------------------------
for (i in training_sets){
  lr = lm(duration ~ ., data = i)
  lr.summary = summary(lr)
  r.squared  = round(lr.summary$r.squared,4)
  Count = Count + 1
  m2.results[Count] <- r.squared
  print(paste('Model =>', Count, 'completed'))
}

# Write Results To File
setwd('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/output')
write.csv(m2.results, 'm2_mlr_r2_datasets_1to6_04202019.csv')
print('hello world')

# Generate Graph of Results
m2.r2 = c(m2.results$Index, m2.results$V2, m2.results$V3, m2.results$V4, m2.results$V5, m2.results$V6)
barplot(m2.r2, names.arg = c('s1_50k', 's2_100k', 's3_250k', 's4_50k.wl', 
                             's5_100k.wl', 's6_250k.wl'), main = 'M2 MLR - ALL DATASETS', 
        xlab = 'Datasets', ylab = 'R2')



# Get RSE:  All Datasets--------------------------------------------------------------------------------
training_sets <- list(s1.50k.nolimits, s4.50k.wlimits)
m2.rse        <- data.frame('Index' = 1)
Count          = 0

for (i in training_sets){
  lr = lm(duration ~ ., data = i)
  lr.summary = summary(lr)
  rse        = sqrt(sum(lr.summary$residuals^2) / length(lr.summary$residuals))
  print(paste('RSE', rse))
  Count = Count + 1
  m2.rse[Count] <- rse
  print(paste('Model =>', Count, 'completed'))
}


# Generate Graph of Resuls
m2.rse_list = c(m2.rse$Index, m2.rse$V2)
barplot(m2.rse_list, names.arg = c('s1_50k', 's4_50k.wl'), main = 'M2 MLR W-DUMMY COLS - ALL DATASETS', 
        xlab = 'Datasets', ylab = 'RSE')





