##################################### Fit DT using Rpart #################################

# we will use another package that fits a DT that allows for a visualization:

library(rpart)

set.seed(seed)

recency_interpretation

DT_rpart <- rpart(
  recency_interpretation ~ .,
  data = train_balanced_trace,
  method = "class", # for classification
  xval = 10 # 10 fold CV
) # default attribute selection measure is Gini Index

DT_rpart # run this to see information about the fitted tree

# Let's obtain the complexity parameter (cp). The complexity parameter is used to control the size of the decision tree and to select the optimal tree size.
# The CP values indicate how much the overall error rate decreases with each split. A large CP indicates that a split resulted in a significant decrease in error, while a smaller CP suggests a less impactful split. This parameter determines a threshold under which the split of a node is not worth the complexity.
# If the cost of adding another variable to the decision tree from the current node is
# above the value of cp, then tree building should not continue.

# The rel error is the total error of the model divided by the error of the initial model (a model with just the root node, predicting the most frequent class). It's a measure of the error relative to the simplest possible model.

# The xerror is the cross-validation error of the model. It is computed during the tree-building process if cross-validation is enabled (e.g., using the xval argument in rpart()). This error is estimated by applying the decision tree to each of the cross-validation folds used during tree construction. It provides a measure of how well the tree is likely to perform on unseen data, hence an estimate of the model's generalization error. Typically, it helps identify if the model is overfitting. If xerror starts to increase as the complexity of the model increases (more splits in the tree), it may suggest that simpler models are preferable.

# The xstd is the standard error of the cross-validation error (xerror). This value provides an indication of the variability of the cross-validation error estimate. A high standard error suggests that the cross-validation error might not be a reliable estimate of the model's error on new data, possibly due to the model being unstable across different subsets of the training data or due to a small number of cross-validation folds.

# rpart() automatically computes the optimal tree size (considering complexity cost) using these metrics. Specifically, xerror and xstd are used to determine the smallest tree that is within one standard error of the minimum cross-validation error (xerror + xstd). This criterion helps to balance model accuracy with complexity, aiming to avoid overfitting while maintaining sufficient explanatory power.

printcp(DT_rpart)
plotcp(DT_rpart)


### Extract predicted probabilities:
# Note: this provides two columns - the predicted probabilities for "0" in column 1 and "1" in column 2.

pred_prob_DT_train <- predict(DT_rpart, newdata = train_balanced_trace, type = "prob")
train_DT_rpart <- cbind(train_balanced_trace, pred_prob_DT_train[, 2, drop = FALSE]) # We only want the probs in column 2 (for "1")


pred_prob_DT_test <- predict(DT_rpart, newdata = test_trace, type = "prob")
test_DT_rpart <- cbind(test_trace, pred_prob_DT_test[, 2, drop = FALSE]) # We only want the probs in column 2 (for "1")


### Then select a cut-off (or find optimal cut-off):

threshold_DT <- 0.5 # specify the cut-off here (this can change to find the optimal)

# training (note the predicted probabilities are now in a column named "1")
train_DT_rpart$pred_class <- factor(ifelse(
  train_DT_rpart$`1` > threshold_DT,
  "1",
  "0"
))

# predicted classes first then actual classes
confusionMatrix(
  train_DT_rpart$pred_class,
  train_DT_rpart$income,
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_DT_train_rpart <- roc(train_DT_rpart$income, train_DT_rpart$`1`)
auc(roc_DT_train_rpart)
plot(roc_DT_train_rpart)

# test
test_DT_rpart$pred_class <- factor(ifelse(
  test_DT_rpart$`1` > threshold,
  "1",
  "0"
))

# predicted classes first then actual classes
confusionMatrix(
  test_DT_rpart$pred_class,
  test_DT_rpart$income,
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_DT_test_rpart <- roc(test_DT_rpart$income, test_DT_rpart$`1`)
auc(roc_DT_test_rpart)
plot(roc_DT_test_rpart)

# visualize the DT:

library(rpart.plot)

dev.new(width = 15, height = 20) # This just allows the plot to be shown in a separate window (useful for small screens)

rpart.plot(DT_rpart)
rpart.plot(DT_rpart, yesno = 1, type = 2, fallen.leaves = FALSE) # add additional options to change the appearance.
# see http://www.milbo.org/rpart-plot/prp.pdf for more options to customize the plot