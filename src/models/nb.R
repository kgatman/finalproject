


############################## Fit Naive Bayes Classifier ##############################
# Build and train the Naive Bayes Classifier (https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/naive-bayes.html):

trace_nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_trace_h2o,
  laplace = 0, # a smoothing parameter for categories with
  nfolds = 5, # for 5-fold CV
  seed = seed
)

# check performance of model:
h2o.performance(trace_nb)

# we will extract the predicted probabilities of class label = 1, append it to the original train and test sets, determine the predicted class for each based on the threshold and then create our confusion matrix and obtain the performance measures:

# Save predicted probabilities
preds_nb_train <- h2o.predict(trace_nb, train_trace_h2o)
preds_nb_test <- h2o.predict(trace_nb, test_trace_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_nb_train <- as.data.frame(preds_nb_train)
preds_nb_test <- as.data.frame(preds_nb_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_nb_pred <- cbind(train_trace, preds_nb_train[, 3, drop = FALSE]) # drop = FALSE keeps the original name of this 3rd column (p1) in the resulting data frame
test_nb_pred <- cbind(test_trace, preds_nb_test[, 3, drop = FALSE])


# Create confusion matrix using a threshold:
threshold <- 0.5

# ---------------------------------------------------------
# 1. TRAINING SET EVALUATION
# ---------------------------------------------------------

# Create predicted class based on threshold
train_nb_pred$pred_class <- factor(ifelse(
  train_nb_pred$p1 > threshold,
  "1",
  "0"
), levels = c("0", "1"))

# Ensure the actual truth is a factor with the same levels
train_nb_pred$recency_interpretation <- factor(
  train_nb_pred$recency_interpretation, 
  levels = c("0", "1")
)

# FIX: Pass 'recency_interpretation' (Truth) instead of 'p1'
confusionMatrix(
  data = train_nb_pred$pred_class,        # Predicted Class
  reference = train_nb_pred$recency_interpretation, # Actual Class
  positive = "1",
  mode = "everything"
)

# ROC and AUC (This requires Numeric inputs, so we calculate it separately)
# We use as.numeric(as.character(...)) to safely convert factors back to numbers if needed, 
# but p1 is already numeric.
roc_nb_train <- roc(train_nb_pred$recency_interpretation, train_nb_pred$p1)
auc(roc_nb_train)
plot(roc_nb_train, main = "ROC Curve - Training Set | naive Bayes")

# ---------------------------------------------------------
# 2. TEST SET EVALUATION
# ---------------------------------------------------------

# Create predicted class based on threshold
test_nb_pred$pred_class <- factor(ifelse(
  test_nb_pred$p1 > threshold, 
  "1", 
  "0"
), levels = c("0", "1"))

# Ensure the actual truth is a factor with the same levels
test_nb_pred$recency_interpretation <- factor(
  test_nb_pred$recency_interpretation, 
  levels = c("0", "1")
)

# FIX: Pass 'recency_interpretation' instead of 'p1'
# REMOVED: The lines that converted recency_interpretation to as.numeric()
confusionMatrix(
  data = test_nb_pred$pred_class, 
  reference = test_nb_pred$recency_interpretation,
  positive = "1",
  mode = "everything"
)

# ROC and AUC for Test
roc_nb_test <- roc(test_nb_pred$recency_interpretation, test_nb_pred$p1)
auc(roc_nb_test)
plot(roc_nb_test, main = "ROC Curve - Test Set | naive Bayes")
