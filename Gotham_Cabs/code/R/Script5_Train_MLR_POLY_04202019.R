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



## M1:    MULTILINEAR REGRESSION_______________________________________________________________________

# Train Model
m3.mlr = lm(duration ~ ., data = s1.50k.nolimits_ran)
m3.summary = summary(m3.mlr)                                            # return summary
m3.summary

# Make Prediction
m3.mlr.predict    = predict(m3.mlr, s1.50k.nolimits_ran)
m3.mlr.rse         = sqrt(sum((s1.50k.nolimits_ran$duration - m3.mlr.predict)^2) / (nrow(s1.50k.nolimits_ran) -2) )
m3.mlr.rse         # Residual Standard Error = 285.62




## M2:    MLR - TEST ALL DATASETS______________________________________________________________________

# Train Model - s1-S6
training_sets <- list(s1.50k.nolimits_ran, s2.100k.nolimits_ran, s3.250k.nolimits_ran, s4.50k.wlimits_ran, 
                      s5.100k.wlimits_ran, s6.250k.wlimits_ran)
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
training_sets <- list(s1.50k.nolimits_ran, s2.100k.nolimits_ran, s3.250k.nolimits_ran, s4.50k.wlimits_ran, 
                      s5.100k.wlimits_ran, s6.250k.wlimits_ran)
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
m2.rse_list = c(m2.rse$Index, m2.rse$V2, m2.rse$V3, m2.rse$V4, m2.rse$V5, m2.rse$V6)

barplot(m2.rse_list, names.arg = c('s1_50k', 's2_100k', 's3_250k', 's4_50k.wl', 
                             's5_100k.wl', 's6_250k.wl'), main = 'M2 MLR - ALL DATASETS', 
        xlab = 'Datasets', ylab = 'RSE')

