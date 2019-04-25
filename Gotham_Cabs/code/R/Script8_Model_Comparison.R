# Clear Namespace
rm(list = ls())

# Create Vector w/ Model RSE Results
#model.names = c('SLR.RSE', 'MLR.RSE', 'MLR.DUMMIES.RSE', 'MLR.POLY.RSE', 'MLR.L1L2.RSE', 'MLR.STEPWISE.RMSE')
#model.rse   = c(343.66, 241.041, 246.85, 244.00, 241.05, 245.80)

model.names = c('SLR.RSE', 'MLR.DUMMIES.RSE', 'MLR.STEPWISE.RMSE', 'MLR.POLY.RSE', 'MLR.L1L2.RSE', 'MLR.RSE')
model.rse   = c(343.66, 246.85, 245.8, 244.00, 241.05, 241.041)


# Create DataFrame
df = data.frame(row.names = model.names)
df$RSE = model.rse


# Generate a Plot for Train & Test Points

ggplot(df, aes(y = df$RSE, x = reorder(model.names, -df$RSE), fill = model.names)) + 
  geom_p(stat = 'identity') + 
  ggtitle('Multilinear Regression - RSE Over 6 Datasets')  +
  scale_y_continuous(breaks = pretty(df$RSE, n = 5))