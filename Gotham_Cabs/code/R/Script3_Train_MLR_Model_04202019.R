## REFERENCES_____________________________________________________________________________
'MLR Output:      https://www.graphpad.com/support/faq/standard-deviation-of-the-residuals-syx-rmse-rsdr/'

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

# Train
s1.train = s1.50k.nolimits_ran[1:  (nrow(s1.50k.nolimits_ran)  * .7), ]
s2.train = s2.100k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s3.train = s3.250k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s4.train = s4.50k.wlimits_ran[1:   (nrow(s4.50k.wlimits_ran)  * .7), ]
s5.train = s5.100k.wlimits_ran[1:  (nrow(s5.100k.wlimits_ran)  * .7), ]
s6.train = s6.250k.wlimits_ran[1:  (nrow(s6.250k.wlimits_ran)  * .7), ]

# Test
s1.test = s1.50k.nolimits_ran[1:  (nrow(s1.50k.nolimits_ran)  * .3), ]
s2.test = s2.100k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .3), ]
s3.test = s3.250k.nolimits_ran[1: (nrow(s3.250k.nolimits_ran) * .3), ]
s4.test = s4.50k.wlimits_ran[1:   (nrow(s4.50k.wlimits_ran)  * .3), ]
s5.test = s5.100k.wlimits_ran[1:  (nrow(s5.100k.wlimits_ran)  * .3), ]
s6.test = s6.250k.wlimits_ran[1:  (nrow(s6.250k.wlimits_ran)  * .3), ]



## M1:    MULTILINEAR REGRESSION_______________________________________________________________________

# Train Model
m3.mlr = lm(duration ~ ., data = s1.50k.nolimits_ran)
m3.summary = summary(m3.mlr)                                            # return summary
m3.summary

# Make Prediction
m3.mlr.predict    = predict(m3.mlr, s1.50k.nolimits_ran)
m3.mlr.ss         = sum((s1.50k.nolimits_ran$duration - m3.mlr.predict)^2)
m3.mlr.mse        = mean((s1.50k.nolimits_ran$duration - m3.mlr.predict)^2)


## M2:    MLR - TEST ALL DATASETS______________________________________________________________________


# Train Model - s1-S6
training_sets <- list(s1.50k.nolimits_ran, s2.100k.nolimits_ran, s3.250k.nolimits_ran, s4.50k.wlimits_ran, 
                      s5.100k.wlimits_ran, s6.250k.wlimits_ran)
m2.results    <- data.frame('Index' = 1)
Count          = 0

for (i in training_sets){
  lr = lm(duration ~ ., data = i)
  lr.summary = summary(lr)
  r.squared  = round(lr.summary$r.squared,4)
  Count = Count + 1
  m2.results[Count] <- r.squared
  print(paste('Model =>', Count, 'completed'))
}

setwd('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/output')
write.csv(m2.results, 'm2_mlr_r2_datasets_1to6_04202019.csv')

print('hello world')



# Train Model 5: Backward Selection
train.control = trainControl(method = 'cv', number = 10)

m5.backward = train(duration ~ ., data = s1.50k.nopp.train, 
                    method = 'leapBackward',                                 # Step selection
                    tuneGrid = data.frame(nvmax = 1:7),                      # Number of features to consider in the model
                    trControl = train.control)                               # Cross Validation technique 
m5.backward$results
summary(m5.backward$finalModel)      # An asterisk specifies that the feature was included in the model


# Train Model 6:  Forward Selection
m6.forward = train(duration ~ ., data = s1.50k.nopp.train, 
                   method = 'leapForward', 
                   tuneGrid = data.frame(nvmax = 1:7), 
                   trControl = train.control)

m6.forward$results
summary(m6.forward$finalModel)     


# Train Model 7:  Stepwise Selection Using Processed Data
train.control = trainControl(method = 'cv', number = 10)
m7.backward = train(duration ~ ., data = s1.50k.pp.train, 
                    method = 'leapBackward', 
                    tuneGrid = data.frame(nvmax = 1:7), 
                    trControl = train.control)
m7.backward$results
summary(m7.backward$finalModel)     

# How can you make a prediction suing train?


# Need to try Ridge & Lasso








