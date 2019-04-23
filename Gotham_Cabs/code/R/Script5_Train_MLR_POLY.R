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

# Calculate Number of Training Observations
train_nrows_50k  = (nrow(s1.50k.nolimits)  * .7)
train_nrows_100k = (nrow(s2.100k.nolimits)   * .7)
train_nrows_250k = (nrow(s3.250k.nolimits)   * .7)


# Train
s1.train = s1.50k.nolimits_ran[1:  (nrow(s1.50k.nolimits_ran)  * .7), ]
s2.train = s2.100k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s3.train = s3.250k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s4.train = s4.50k.wlimits_ran[1:   (nrow(s4.50k.wlimits_ran)  * .7), ]
s5.train = s5.100k.wlimits_ran[1:  (nrow(s5.100k.wlimits_ran)  * .7), ]
s6.train = s6.250k.wlimits_ran[1:  (nrow(s6.250k.wlimits_ran)  * .7), ]

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
  m1.mlr.poly = lm(duration ~ poly(pickup_x, pickup_y, dropoff_x, dropoff_y, weekday, hour_, day_, distance, month_, speed, 
                                   degree = polynomial, raw = T), data = data_train)
  # Return Summary Stats
  if (objective == 'train'){
    m1.summary = summary(m1.mlr.poly)
    return(m1.summary)
  }
    # Generate Prediction
    if (objective == 'test'){ 
      m1.mlr.poly.predict    = predict(m1.mlr.poly, data_test)
      m1.mlr.poly.rse        = sqrt(sum((s4.test$duration - m1.mlr.poly.predict)^2) / (length(m1.mlr.poly.predict) -2))
      return(m1.mlr.poly.rse)
    }
}


#mlr.poly(s4.train, s4.test, 4, 'train')


for (i in seq(1,4)){
  rse = mlr.poly(s4.train, s5.test, i, 'test')
  print(paste('Polynomial =>', i, 'RSE TEST =>', rse))
}


# Aggregate Results - Create Graph
'Below results were typed out using the above functions'
num_poly     = c(1,2,3,4)
d1_rse_train = c(280, 196, 149, 113)
d1_rse_test  = c(737, 766, 801, 3438)
d2_rse_train = c(246, 153, 106, 81)
d2_rse_test  = c(247, 156, 262, 706)
d3_rse_test  = c(733, 781, 19394, 1779816)

# Dataset 1
d1_results <- matrix(c(280, 196, 149, 113, 737, 766, 801, 3438), ncol=2, byrow = TRUE)
colnames(d1_results) = c('d1_train', 'd1_test')
rownames(d1_results) = c('p1', 'p2', 'p3', 'p4')
d1_plot = barplot(d1_results, beside = T, legend = rownames(training_results), panel.first = grid(),
                  main = 'MLR RESULTS - POLYNOMIALS 1-4', 
                  ylab = 'TRAIN - RSE', 
                  xlab = 'DATASETS')

# Dataset 2
d2_results <- matrix(c(246, 153, 106, 81, 733, 781, 1500, 2000), ncol = 2, byrow = TRUE)
colnames(d2_results) = c('d2_train', 'd2_test')
rownames(d2_results) = c('p1', 'p2', 'p3', 'p4')
d2_plot = barplot(d2_results, beside = T, legend = rownames(d2_results), panel.first = grid(),
                  main = 'MLR RESULTS - POLYNOMIALS 1-4', 
                  ylab = 'TRAIN - RSE', 
                  xlab = 'DATASETS')





