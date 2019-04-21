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



## M1:    MULTILINEAR REGRESSION - WITH POLYNOMIALS______________________________________________________

# Train Model
m1.mlr.poly = lm(duration ~ poly(pickup_x, pickup_y, dropoff_x, dropoff_y, weekday,hour_, day_,distance,month_, speed , 
                                 degree = 2, raw = T), data = s6.train)    # Polynomial up to 10
m1.summary = summary(m1.mlr.poly)
m1.summary

# Make Prediction
m1.mlr.poly.predict    = predict(m1.mlr.poly, s6.test)
m1.mlr.poly.rse        = sqrt(sum((s6.test$duration - m1.mlr.poly.predict)^2) / (length(m1.mlr.poly.predict) -2) )
m1.mlr.poly.rse        # Residual Standard Error = 107.89



## M2:    MLR - TEST ALL DATASETS______________________________________________________________________


# Get RSE:  All Datasets--------------------------------------------------------------------------------
training_sets <- list(s1.train, s2.train, s3.train, s4.train, s5.train, s6.train)
test_sets     <- list(s1.test, s2.test, s3.test, s4.test, s5.test, s6.test)
m2.rse.train  <- data.frame('Index' = 1)
m2.rse.test   <- data.frame('Index' = 1)

Count          = 1

for (i in training_sets){
  # Train Test Model
  print(paste('Training Model', Count))
  m1.mlr.poly         = lm(duration ~ poly(pickup_x, pickup_y, dropoff_x, dropoff_y, weekday,hour_, day_, distance,month_, speed, 
                           degree = 4, raw = T), data = i) 
  m1.summary          = summary(m1.mlr.poly)
  rse.train           = sqrt(sum(m1.summary$residuals^2) / length(m1.summary$residuals))
  m2.rse.train[Count] = rse.train 
  print(paste('Generate rse-train for', Count, 'RSE =>', rse.train))
}
'  
  for (j in test_sets){  
    # Generate Prediction
    print(paste('Generating Prediction for', Count))
    m1.mlr.poly.predict    = predict(m1.mlr.poly, j)
    m1.mlr.poly.rse        = sqrt(sum((j$duration - m1.mlr.poly.predict)^2) / (nrow(m1.mlr.poly.predict) -2) )
    m2.rse.test[Count]     = m1.mlr.poly.rse        
  
    # Increment Count & Return Feedback
    Count = Count + 1
    print(paste('Model =>', Count, 'completed'))
    }
}
'

m2.rse.train
m2.rse.test

# Generate Graph of Resuls
m2.rse_list = c(m2.rse$Index, m2.rse$V2, m2.rse$V3, m2.rse$V4, m2.rse$V5, m2.rse$V6)

barplot(m2.rse_list, names.arg = c('s1_50k', 's2_100k', 's3_250k', 's4_50k.wl', 
                             's5_100k.wl', 's6_250k.wl'), main = 'M2 MLR - ALL DATASETS', 
        xlab = 'Datasets', ylab = 'RSE')

