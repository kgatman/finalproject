

###################### Combination of over and under #################################

# We can apply a combination of both over- and under-sampling, where the number of minority
# cases increases and the number of majority cases decreases.

total_both <- nrow(train_depression) # specify the total sample size after the procedure
fraction_depression_new <- 0.50 # specify the approx proportion of minority cases to be produced

train_both <- ovun.sample(
  Depression ~ .,
  data = train_depression,
  method = "both",
  N = total_both,
  p = fraction_depression_new,
  seed = seed
)

# Extract and save the resulting under-sampled data (list):
train_both_data <- train_both$data
summary(train_both_data$Depression)

####################################### SMOTE ##################################

install.packages("UBL")
install.packages("sf") # a dependency for the UBL package
library(UBL)

# We use the SmoteClassif function which allows us to specify the method of determining the synthetic observations based on the nearest neighbours. We use the dist option to specify the method to use based on the type of data (see https://rdrr.io/cran/UBL/man/smoteClassif.html)

# The depression data has mixed attributes (numerical and categorical), so we use HEOM or HVDM

# We specify C.perc = "balanced" to automatically balance the data based on the originalsize. Or we can specify a named list of percentage for each class.

# In this example, we keep the number of majority cases the same by specifying No=1, but we increase the number of minority to approx the same as that of the majority by dividing the number of majority cases by the number of minority cases (we save this calculation in an object called Minority_perc below)

Minority_perc <- nrow(train_depression[train_depression$Depression == "No", ]) /
  nrow(train_depression[train_depression$Depression == "Yes", ])

set.seed(seed)
train_smote_data = SmoteClassif(
  Depression ~ .,
  train_depression,
  C.perc = list(No = 1, Yes = Minority_perc),
  k = 4, # number of nearest neighbours,
  dist = "HVDM"
)

summary(train_smote_data$Depression)