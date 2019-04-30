source('/home/ccirelli2/Desktop/Repositories/ML_Final_Project_2019/Gotham_Cabs/code/R/Decision_Tree/module0_random_forest.R')

# Test Function
f.test = function(){
  print('hello world')
}




# Random Forest - Test Number of Trees

rf_num_trees = function(data.train, data.test, ntrees.from, ntrees.to, ntrees.step, list.ntrees, list.oob.rse, list.test.rse, Count){
  
  for (i in seq(ntrees.from, ntrees.to, ntrees.step)){
    # Train Model
    print('Training Model')
    m0 = ranger(duration ~., data = data.train, num.trees = i)
    # Generate OOB RSE
    print('Generating OOB RSE')
    m0.oob.rse            = sqrt(m0$prediction.error)
    list.oob.rse[Count]   <<- m0.oob.rse
    # Generate Prediction Using New Sample Data
    print('Generating Test Prediction')
    m0.predict            = predict(m0, data.test)
    # Calculate Test RSE
    print('Generating Test RSE')
    m0.test.rse           = sqrt(sum((data.test$duration - m0.predict$predictions)^2) / (length(m0.predict$predictions)-2))
    list.test.rse[Count]  <<- m0.test.rse
    # Increase Count
    Count                 <<- Count + 1
    # Return Model
    print('Model Completed.  Returning model object to user')
    return(m0)
  }
}
  
  
  return(m0)
}

