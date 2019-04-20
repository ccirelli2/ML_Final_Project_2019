/*How can we represent distance using these coordinates. 

Derived values:
- Day of week, 
- Day of month
- Hour of day (1-24)
- Distance 

*/


SELECT 	
		                    
      pickup_x, 
      pickup_y, 
      dropoff_x, 
      dropoff_y, 
      COUNT(duration) AS Route_Count,
      ROUND(AVG(duration),0) AS AVG_Duration 
      
FROM GSU.ML_FinProj_GothamCab_Train
GROUP BY pickup_x, pickup_y, dropoff_x, dropoff_y 
ORDER BY COUNT(duration) DESC
LIMIT 20;