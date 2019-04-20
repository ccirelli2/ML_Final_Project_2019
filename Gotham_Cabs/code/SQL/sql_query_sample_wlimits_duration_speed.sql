SELECT 
		duration, 
        pickup_x, 
        pickup_y, 
        dropoff_x, 
        dropoff_y, 
        weekday, 
        hour_, 
        day_, 
        distance, 
        month_, 
        speed

FROM GSU.ML_FinProj_GothamCab_Train

WHERE speed IS NOT NULL
AND duration != 0
AND duration < 5000
AND speed < 1000
ORDER BY RAND()
LIMIT 100000;
