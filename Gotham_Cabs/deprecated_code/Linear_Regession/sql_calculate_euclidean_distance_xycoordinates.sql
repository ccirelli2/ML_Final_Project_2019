UPDATE GSU.ML_FinProj_GothamCab_Train
SET Distance = ROUND(SQRT(POWER(dropoff_x - pickup_x,2) + POWER(dropoff_y - pickup_Y,2)),2);