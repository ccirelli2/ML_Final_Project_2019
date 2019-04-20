SELECT 
		*

FROM GSU.ML_FinProj_GothamCab_Train

WHERE speed IS NOT NULL
AND duration != 0
AND duration < 5000
AND speed < 1000
ORDER BY RAND()
LIMIT 10000;
