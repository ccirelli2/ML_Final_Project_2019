# MLR - APPLY RIDGE & LASSO

## DOCUMENTATION__________________________________________________________________________
'
1.) Should we regularize our target?
    https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
'


## CLEAR NAMESPACE________________________________________________________________________
rm(list = ls()) 

## IMPORT LIBRARIES_______________________________________________________________________
library(lattice)
library(ggplot2)
library(caret)  # used for parameter tuning
library(glmnet)
library(pls)
library(ISLR)


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

# Calculate Number of Training Observations
train_nrows_50k  = (nrow(s1.50k.nolimits)  * .7)
train_nrows_100k = (nrow(s2.100k.nolimits)   * .7)
train_nrows_250k = (nrow(s3.250k.nolimits)   * .7)

# Train
s1.train = s1.50k.nolimits_ran[1:  (nrow(s1.50k.nolimits_ran)  * .7), ]
s2.train = s2.100k.nolimits_ran[1: (nrow(s2.100k.nolimits_ran) * .7), ]
s3.train = s3.250k.nolimits_ran[1: (nrow(s3.250k.nolimits_ran) * .7), ]
s4.train = s4.50k.wlimits_ran[1:   (nrow(s4.50k.wlimits_ran)  * .7), ]
s5.train = s5.100k.wlimits_ran[1:  (nrow(s5.100k.wlimits_ran)  * .7), ]
s6.train = s6.250k.wlimits_ran[1:  (nrow(s6.250k.wlimits_ran)  * .7), ]

# Test
s1.test = s1.50k.nolimits_ran[ train_nrows_50k:   nrow(s1.50k.nolimits_ran), ] # Index from training to total
s2.test = s2.100k.nolimits_ran[train_nrows_100k:  nrow(s2.100k.nolimits_ran), ]
s3.test = s3.250k.nolimits_ran[train_nrows_250k:  nrow(s3.250k.nolimits_ran), ]
s4.test = s4.50k.wlimits_ran[  train_nrows_50k:   nrow(s4.50k.wlimits_ran), ]
s5.test = s5.100k.wlimits_ran[ train_nrows_100k:  nrow(s5.100k.wlimits_ran), ]
s6.test = s6.250k.wlimits_ran[ train_nrows_250k:  nrow(s6.250k.wlimits_ran), ]


# Separate Target & Feature Values
s1_y = s1.50k.nolimits_ran$duration
s1_x = as.matrix(s1.50k.nolimits_ran[,2:11])

s2_y = s2.100k.nolimits_ran$duration
s2_x = as.matrix(s2.100k.nolimits_ran[,2:11])

s3_y = s3.250k.nolimits_ran$duration
s3_x = as.matrix(s3.250k.nolimits_ran[,2:11])

s4_y = s4.50k.wlimits_ran$duration
s4_x = as.matrix(s4.50k.wlimits_ran[,2:11])

s5_y = s5.100k.wlimits_ran$duration
s5_x = as.matrix(s5.100k.wlimits_ran[,2:11])

s6_y = s6.250k.wlimits_ran$duration
s6_x = as.matrix(s6.250k.wlimits_ran[,2:11])

# Generate Grid Possible Values Lambda
grid = 10^seq(from = 10, to = -2, length = 100)                 #length = desired length of sequence


# M1:   TRAIN RIDGE & LASSO MODELS - COMPARE LAMBDAS__________________________________________

# Train Model 
m1.ridge <- glmnet(s1_x, s1_y, alpha = 0, lambda = grid, standardize = TRUE)

# Create List to Capture Sum of Coefficients
list_sum_ceoffs <- c()
df = data.frame(row.names = grid)

# Compare Value of Coefficients 
for (i in seq(1,length(m1.ridge$lambda))){
  sum_coeff = sum((sqrt(coef(m1.ridge)[,i]^2)))
  list_sum_ceoffs[i] = sum_coeff
}

# Plot Sum of Squared Coefficients For Each Value of Lambda
df$sumcoeff = list_sum_ceoffs
ggplot(data = df, aes(x = seq(1,100), y = df$sumcoeff)) + geom_line() + ggtitle('Ridge - Sum of Squared Coefficeints For Each Lambda')





# M2:   FIND OPTIMAL LAMBDAS FOR RIDGE & LASSO________________________________________________

# Define Lists to Capture Output
rse.test = c()
list.lambda    = c()
X.datasets     = list(s1_x, s4_x, s5_x, s6_x)
Y.datasets     = list(s1_y, s4_y, s5_y, s6_y)
names.datasets = c('50knl','50kwl', '100kwl', '250kwl')

for (i in seq(1,4)){
  # Create Data Objects
  print('Creating Datasets')
  X = X.datasets[[i]]
  Y = Y.datasets[[i]]
  # Train Model Using CV
  print('Training CV Model')
  m_cv = cv.glmnet(X, Y, alpha = 0, lambda = grid, standardize = TRUE, nfolds = 10)
  # Get Best Lambda
  cv_lambda = m_cv$lambda.min
  # Fit Model w/ Best Lambda
  print('Fit Model w/ Best Lambda')
  m_optimal <- glmnet(X, Y, alpha = 0, lambda = cv_lambda, standardize = TRUE)
  # Generate Prediction
  print('Generate Prediction')
  y_hat_cv <- predict(m_optimal, X)
  # Calculate RSE
  print('Calculate RSE')
  model_cv_rse = sqrt(sum((Y - y_hat_cv)^2) / (length(Y) - 2))
  print(paste('Model ', i, 'RSE =>', model_cv_rse))
  # Append RSE Values To List
  rse.test[i] = round(model_cv_rse,0)
  print(paste('Iteration', i, 'completed'))
}



# Create DataFrame
df = data.frame(row.names = names.datasets)
df$ridge.rse = rse.test


# Generate a Plot for Train & Test Points
ggplot(df, aes(y = df$ridge.rse, x = names.datasets, fill = names.datasets)) + geom_bar(stat = 'identity') + 
  ggtitle('Multilinear Lasso Regression - 4 Datasets - RSE') + 
  scale_y_continuous(breaks = pretty(df$ridge.rse, n = 5))
df




















