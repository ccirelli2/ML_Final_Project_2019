




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








