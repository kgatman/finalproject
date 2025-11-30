######################################## Neural Network #######################################

# The H2O package supports deep learning and hyperparameter tuning for the NN. It uses the stochastic gradient descent (SDG) optimizer (weights are updated in batches of 1, i.e. after each training example).

# See https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html for all info. 


######################## Convert train/test sets into H2O data frames ######################


train_h2o <- as.h2o(train_processed)
test_h2o <- as.h2o(test_processed)


######################### Specify name of target and predictors #########################

target <- "recency_interpretation"

predictors <- setdiff(names(train_processed), target)

############### Train a basic NN for tuning the hyperparameters ##################

# Set hyperparameters for grid search
hyper_params <- list(
  hidden = list(c(5, 5), c(10, 10), c(5, 10, 5)),  # 2 hidden layers with 5 nodes each, then 2 with 10 nodes each, then 3 with 5, 10 and 5 nodes..
  activation = c("Rectifier", "Tanh", "Maxout"),  # activation functions (maxout is the max of the inputfor the node, which is the weighted outputs from the previous node plus the bias)
  #epochs = c(10, 20),  # number of epochs - uncomment if included in grid search
  rate = c(0.001, 0.01)  # learning rate seq(0.00001,0.1, length=0.)
)


# Perform grid search for hyperparameter tuning
grid_search <- h2o.grid(
  algorithm = "deeplearning", 
  grid_id = "nn_grid",
  hyper_params = hyper_params,
  x = predictors,
  y = target,
  standardize = FALSE, # this has already been done
  training_frame = train_h2o,
  search_criteria = list(strategy = "Cartesian"),
  adaptive_rate = FALSE, # turn on and off
  nfolds = 10, # Cross validation 10-folds, change to 5 for 5-fold CV
  stopping_rounds = 0, # turn off early stopping
  seed = seed
)



# View grid sorted by logloss (change metric if needed)
grid_results <- h2o.getGrid(grid_id = "nn_grid", 
                            sort_by = "logloss", 
                            decreasing = FALSE)
print(grid_results)

# extract best model
best_model <- h2o.getModel(grid_results@model_ids[[1]])

best_params <- best_model@allparameters
print(best_params)

NN <- h2o.deeplearning(
  x = predictors,
  y = target,
  training_frame = train_h2o,  # Combine training + validation if needed
  hidden = best_params$hidden,
  activation = best_params$activation,
  rate = best_params$rate,
  adaptive_rate = FALSE,
  epochs = best_params$epochs, # defaults to 10 if not included in hyper_params
  seed = seed
) #ignore the error regarding constant columns

h2o.performance(NN)

# Save predicted probabilities
preds_NN_train <- h2o.predict(NN, train_h2o)
preds_NN_test <- h2o.predict(NN, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_NN_train <- as.data.frame(preds_NN_train)
preds_NN_test <- as.data.frame(preds_NN_test)

train_NN_pred <- cbind(train_processed, preds_NN_train[, "X1", drop = FALSE]) # drop = FALSE keeps the original name of this column (yes) in the resulting data frame
test_NN_pred <- cbind(test_processed, preds_NN_test[, "X1", drop = FALSE])

# Create confusion matrix using a threshold:

threshold <- 0.5

# training

train_NN_pred$pred_class <- factor(ifelse(train_NN_pred$X1 > threshold,"X1","X0"))

# predicted classes first then actual classes
caret::confusionMatrix(
  train_NN_pred$pred_class,
  train_NN_pred$recency_interpretation,
  positive = "X1",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_NN_train <- roc(train_NN_pred$recency_interpretation, train_NN_pred$X1)
auc(roc_NN_train)
plot(roc_NN_train)

# test
test_NN_pred$pred_class <- factor(ifelse(test_NN_pred$X1 > threshold, "X1", "X0"))

# predicted classes first then actual classes
caret::confusionMatrix(
  test_NN_pred$pred_class,
  test_NN_pred$recency_interpretation,
  positive = "X1",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_NN_test <- roc(test_NN_pred$recency_interpretation, test_NN_pred$X1)
auc(roc_NN_test)
plot(roc_NN_test)

### Variable importance for the NN

var_imp_NN <- h2o.varimp(NN)
View(var_imp_NN)
h2o.varimp_plot(NN)



h2o.shutdown(prompt = FALSE)
