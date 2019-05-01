# Clear Namespace
rm(list = ls())

# Create Vector w/ Model RSE Results
#model.names = c('SLR.RSE', 'MLR.RSE', 'MLR.DUMMIES.RSE', 'MLR.POLY.RSE', 'MLR.L1L2.RSE', 'MLR.STEPWISE.RMSE')
#model.rse   = c(343.66, 241.041, 246.85, 244.00, 241.05, 245.80)

model.names = c('SLR', 'MLR', 'Poly', 'L1L2', 'Stepwise')
model.rse   = c(346.6, 336.2, 356.8, 337.9, 339.9)


# Create DataFrame
df = data.frame(row.names = model.names)
df$RSE = model.rse


# Generate a Plot for Train & Test Points

ggplot(df, aes(y = df$RSE, x = reorder(model.names, -df$RSE), fill = model.names)) + 
  geom_bar(stat = 'identity') + 
  ggtitle('COMPARISON REGRESSION APPROACHS')  + xlab('MODELS') + ylab('RMSE') +
  scale_y_continuous(breaks = pretty(df$RSE, n = 5))