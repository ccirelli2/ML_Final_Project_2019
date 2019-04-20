SELECT 

MONTH(pickup_datetime) AS 'MONTH', 
ROUND(AVG(duration),0) AS 'AVERAGE_DURATION'

FROM GSU.ML_FinProj_GothamCab_Train

GROUP BY MONTH(pickup_datetime)
ORDER BY ROUND(AVG(duration),0);