library(ROSE)

###################### Combination of over and under #################################

# We can apply a combination of both over- and under-sampling, where the number of minority
# cases increases and the number of majority cases decreases.

total_both <- nrow(train_trace) # specify the total sample size after the procedure
# total_both = 11 681
fraction_depression_new <- 0.50 # specify the approx proportion of minority cases to be produced

train_both <- ovun.sample(
  recency_interpretation ~ .,
  data = train_trace,
  method = "both",
  N = total_both,
  p = fraction_depression_new,
  seed = seed
)

# Extract and save the resulting under-sampled data (list):
train_both_data <- train_both$data
summary(train_both_data$recency_interpretation)

#renaming to something i'll remember
train_balanced_trace <- train_both_data