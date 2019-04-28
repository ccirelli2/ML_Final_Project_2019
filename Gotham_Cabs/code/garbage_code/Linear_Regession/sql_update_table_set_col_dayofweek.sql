UPDATE GSU.ML_FinProj_GothamCab_Train
SET Weekday = DAYOFWEEK(DATE(pickup_datetime));	