# DATA INSPECTION
'Content:
  1.) Average Duration:     By month, day and route.
                            Plot:  Bar
                            Plot:  Boxplot. 

      Observations:         
                            Duration: See plot.  It appears that the majority are 
                            Month:    Only 6 months worth of data (Jan-June)
                                      There appears to be a gradual increase in duration 
                                      from months 1-6. 

                            Day:      Day 2, 6, 8, 10, 12, 14, 16, 18, 20, 22... not present. 
                                      No decernible relationship on a monthly basis.
                                      Check to see if we can map these dates to days of the week. 
                            
      Definitions:          Boxplot   Lower bar is Q1. 25% percentile. 
                                      Bold bar in middle is the median. 
                                      Box represents the "inner quartile". 
                                      Upper part of box is 3Q or 75% quartile. 

  2.) Route Frequency       Routes most frequently traveled. 
'

# CLEAR NAMESPACE
rm(list = ls())

# LOAD LIBRARIES
library(RMySQL)
install.packages("psych")
library(psych)

# SETUP CONNECTION TO DB
mydb <- dbConnect(RMySQL::MySQL(), user='ccirelli2', 
                  password='Work4starr', dbname='GSU', 
                  host = "127.0.0.1")

# Get List of Tables
dbListTables(mydb)

# QUERIES (1-N)---------------------------------------------------------------------

# Query 1:  All Data
query1_alldata = dbSendQuery(mydb, '
                       SELECT 
                       *
                      
                        FROM ML_FinProj_GothamCab_Train
                        WHERE duration != 0
                        LIMIT 100000')
result_q1 = fetch(query1_alldata, n = -1)

describe.by(result_q1)

# Plot Durations
plot(result_q1$duration)
boxplot(result_q1$duration)
hist(result_q1$duration)
d = density(result_q1$duration)
plot(d)
  
# Query 2:  Average Duration By Month
query2_rel_month_duration = dbSendQuery(mydb, '
                        SELECT 
                        
                        MONTH(pickup_datetime) AS "MONTH", 
                        ROUND(AVG(duration),0) AS "AVERAGE_DURATION"
                        FROM GSU.ML_FinProj_GothamCab_Train

                        GROUP BY MONTH(pickup_datetime)
                        ORDER BY ROUND(AVG(duration),0);')

result_q2 = fetch(query2_rel_month_duration, n = -1)
barplot(result_q2$AVERAGE_DURATION, names.arg = result_q2$MONTH, 
        main = "Avg Duration By Month",
        xlab = "Month", 
        ylab = "Duration")
  


# Query 3:  Relationship of Average Duration By Day of Week
query3_rel_day_duration = dbSendQuery(mydb, '
                        SELECT 

                        DAY(pickup_datetime) AS "DAY", 
                        ROUND(AVG(duration),0) AS "AVERAGE_DURATION"

                        FROM GSU.ML_FinProj_GothamCab_Train

                        GROUP BY DAY(pickup_datetime)
                        ORDER BY DAY(pickup_datetime);')

result_q3 = fetch(query3_rel_day_duration, n = -1)

barplot(result_q3$AVERAGE_DURATION, 
        names.arg = result_q3$DAY, 
        main = "Average Duration By Day Of Week", 
        xlab = "Day",
        ylab = "Duration")


# Query 4:  Relationship of Average Duration By Day of Week
' MySQL DAYOFWEEK:    1 = Sunday
                      7 = Saturday'

query4_rel_weekday_duration = dbSendQuery(mydb, '
                        SELECT 

                        Weekday, 
                        ROUND(AVG(duration),0) AS "AVERAGE_DURATION"

                        FROM GSU.ML_FinProj_GothamCab_Train

                        GROUP BY Weekday
                        ORDER BY Weekday;')

result_q4 = fetch(query4_rel_weekday_duration, n = -1)

barplot(result_q4$AVERAGE_DURATION,
        names.arg = result_q4$Weekday, 
        main = "Average Duration By Weekday", 
        xlab = "Weekday",
        ylab = "Duration")

# Query 8:    Route Duration By Hour of Day (Add Frequency By Day, Duration By Day)
query8_rel_dur_hour_of_day = dbSendQuery(mydb, '
                      SELECT  
                      hour_,
                      ROUND(AVG(duration),0) AS "AVERAGE_DURATION"

                      FROM GSU.ML_FinProj_GothamCab_Train

                      GROUP BY hour_
                      ORDER BY RAND()
                      LIMIT 10000;')
result_q8 = fetch(query8_rel_dur_hour_of_day, n = -1)
plot(result_q8, 
     main = 'Average Duration By Hour of Day', 
     xlab = 'Hour of Day (0-24)', 
     ylab = 'Average Duration')


# Query 5:    Most Traveled Routes
query5_rel_route_freq = dbSendQuery(mydb, '
      SELECT 	
		                    
      pickup_x, 
      pickup_y, 
      dropoff_x, 
      dropoff_y, 
      COUNT(duration) AS Route_Count
       
      FROM GSU.ML_FinProj_GothamCab_Train
      GROUP BY pickup_x, pickup_y, dropoff_x, dropoff_y 
      ORDER BY COUNT(duration) DESC
      LIMIT 20;
                             ')
result_q5 = fetch(query5_rel_route_freq, n = -1)
#setwd('/home/ccirelli2/Desktop/GSU/2019_Spring/ML_Course/Final_Project/Gotham_Cabs/Preliminary_Analysis')
#write.table(result_q5, 'Top_20_Most_Traveled_Results.xlsx') 


# Query 7:    Distance vs Duration
query7_duration_vs_distance = dbSendQuery(mydb, '
      SELECT 	
		                    
      duration, 
      distance
       
      FROM GSU.ML_FinProj_GothamCab_Train
      ORDER BY RAND()
      LIMIT 10000;
                             ')
result_q7 = fetch(query7_duration_vs_distance, n = -1)
plot(log(result_q7$distance), log(result_q7$duration), main = 'Distance vs Duration', xlab = 'Distance', ylab = 'Duration', col=c('red', 'blue'))
 








