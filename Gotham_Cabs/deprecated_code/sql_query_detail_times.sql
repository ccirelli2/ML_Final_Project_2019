SELECT 
		YEAR(pickup_datetime) AS 'YEAR',
        MONTH(pickup_datetime) AS 'MONTH',
        DAY(pickup_datetime) AS 'DAY', 
        pickup_x, 
        pickup_y, 
        dropoff_x, 
        dropoff_y, 
        duration

FROM GSU.ML_FinProj_GothamCab_Train
WHERE duration > 0;
        
    
