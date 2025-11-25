######################### Fit a logistic regression model ################################

# A logistic regression is in the class of a generalized linear model (GLM), various GLMs can be fitted in h2o for different types of responses (continuous, binary, count, multiple categories - multi-class classification)

# Fit the logistic regression model
trace_LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_trace_h2o,
  family = "binomial", # logistic regression
  lambda = 0, # no regularization (like classical GLM)
  compute_p_values = TRUE, # optional: get p-values
  remove_collinear_columns = TRUE
)

# extract p-values for inference:

df <- trace_LR@model[["coefficients_table"]]

options(scipen = 999)

df$OR <- exp(df[,2])

df$p_value <- round(df$p_value,4)

view(df)

# Save predicted probabilities
preds_LR_train <- h2o.predict(trace_LR, train_trace_h2o)
preds_LR_test <- h2o.predict(trace_LR, test_trace_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_LR_train <- as.data.frame(preds_LR_train)
preds_LR_test <- as.data.frame(preds_LR_test)

# Append column 3 (predicted probabilities for class lablel = 1) to original training and test sets:

train_LR_pred <- cbind(train_trace, preds_LR_train[, 3, drop = FALSE]) # drop = FALSE keeps the original name of this 3rd column (p1) in the resulting data frame
test_LR_pred <- cbind(test_trace, preds_LR_test[, 3, drop = FALSE])


# Create confusion matrix using a threshold:

threshold <- 0.5

# training
#train_LR_pred$pred_class <- factor(ifelse(
#  train_LR_pred$p1 > threshold,
#  "1",
#  "0"
#))

train_LR_pred$pred_class <- factor(ifelse(
  train_LR_pred$p1 > threshold,
  "1",
  "0"
), levels = c("0", "1"))


# predicted classes first then actual classes
#confusionMatrix(
#  train_LR_pred$pred_class,
#  train_LR_pred$income,
#  positive = "1",
#  mode = "everything"
#)

confusionMatrix(
  data = train_LR_pred$pred_class,        # Predicted Class
  reference = train_LR_pred$recency_interpretation, # Actual Class
  positive = "1",
  mode = "everything"
)



# actual classes first then predicted probabilities
roc_LR_train <- roc(train_LR_pred$pred_class, train_LR_pred$p1)
auc(roc_LR_train)
plot(roc_LR_train, main = "ROC Curve - Training Set | Logistic Regression")

# test
test_LR_pred$pred_class <- factor(ifelse(test_LR_pred$p1 > threshold, "1", "0"))

# predicted classes first then actual classes
#confusionMatrix(
#  test_LR_pred$pred_class,
#  test_LR_pred$p1,
#  positive = "1",
#  mode = "everything"
#)

confusionMatrix(
  data = test_LR_pred$pred_class,        # Predicted Class
  reference = test_LR_pred$recency_interpretation, # Actual Class
  positive = "1",
  mode = "everything"
)


# actual classes first then predicted probabilities
roc_LR_test <- roc(test_LR_pred$pred_class, test_LR_pred$p1)
auc(roc_LR_test)
plot(roc_LR_test, main = "ROC Curve - Test Set | Logistic Regression")

preds_lr_h2o <- h2o.predict(trace_LR, test_trace_h2o)
preds_lr_df <- as.data.frame(preds_lr_h2o)

roc_LR_test <- roc(
  response = test_trace$recency_interpretation,
  predictor = preds_lr_df$p1,
  levels = c("0", "1")
)

#################### Combine ROC curves of test set for all models ############

# 1. Load necessary libraries
library(rpart)
library(pROC)

# 2. Train the Decision Tree (rpart)
# We use the standard R dataframes (train_trace), NOT the h2o ones
dt_model <- rpart(
  recency_interpretation ~ ., 
  data = train_trace, 
  method = "class"
)

# 3. Predict probabilities on the Test Set
dt_preds <- predict(dt_model, test_trace, type = "prob")

# 4. Create the missing 'roc_DT_test_rpart' object
# We grab column "1" to get the probability of the positive class
roc_DT_test_rpart <- roc(
  response = test_trace$recency_interpretation,
  predictor = dt_preds[, "1"],
  levels = c("0", "1")
)

# -------------------------------------------------------
# 5. NOW RUN THE PLOT
# -------------------------------------------------------

# Plot Base (Naive Bayes)
plot(
  roc_nb_test,
  col = "#458B74",
  lwd = 2,
  main = "naive Bayes vs Decision Tree vs Logistic Regression"
)

# Add Decision Tree (Red) -> This will work now!
lines(roc_DT_test_rpart, col = "#CD3333", lwd = 2)

# Add Logistic Regression (Blue)
# (Assuming roc_LR_test exists from the previous step)
lines(roc_LR_test, col = "#009ACD", lwd = 2)

# Add Legend
legend(
  "bottomright",
  legend = c("Naive Bayes", "Decision tree", "Logistic regression"),
  col = c("#458B74", "#CD3333", "#009ACD"),
  lwd = 2
)


############## Shut down H2O cluster so it doesn't use up any more resources ############

h2o.shutdown(prompt = FALSE)