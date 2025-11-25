##### Splitting The Data

seed <- 123 # set a seed for reproducibility
set.seed(seed)

split <- sample.split(traceData_ready$recency_interpretation, SplitRatio = 0.7)

train_trace <- subset(traceData_ready, split == "TRUE")
test_trace <- subset(traceData_ready, split == "FALSE")




######################## Convert train/test sets into H2O data frames ######################

# To use data in H2O functions/models, it needs to be an H2O data frame. The following converts the built-in R data frame iris into an H2O frame and stores it in the H2O memory space. Now, all future processing (cleaning, modeling, predictions) happens in H2O's memory space (inside the Java engine, not R’s memory). H2O is great at handling big datasets relative to RAM size due to its optimized data structures.

train_trace$recency_interpretation <- factor(train_trace$recency_interpretation, ordered = FALSE)
test_trace$recency_interpretation <- factor(test_trace$recency_interpretation, ordered = FALSE)

train_trace_h2o <- as.h2o(train_trace)
test_trace_h2o <- as.h2o(test_trace)



target <- "recency_interpretation"
predictors <- setdiff(names(train_trace), target)