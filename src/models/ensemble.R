##########################################################################################
#                                                                                        #
#                        H2O Package for  Ensemble methods:                              #
#      Random forests, Gradient Boosted Machines (GBM), Stacked Ensembles                #  
#                                                                                        #
##########################################################################################


################################# Load additional packages ##############################

library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting into train and test sets
library(caret) # used for performance metric functions
library(pROC) # used for obtaining AUC

# Helper function to convert recency_interpretation to factor with levels "0" and "1"
# Handles various input formats: numeric (0/1), character ("0"/"1"), factor, etc.
convert_recency_to_factor <- function(x) {
  orig_vals <- x
  # If it's a factor, get the underlying values
  if(is.factor(orig_vals)) {
    orig_vals <- as.character(orig_vals)
  }
  # Convert to numeric first (handles "0", "1", 0, 1, etc.)
  numeric_vals <- suppressWarnings(as.numeric(orig_vals))
  if(any(is.na(numeric_vals))) {
    # If conversion to numeric failed, check for text values
    # Map "Recent" or anything containing "1" to "1", everything else to "0"
    char_vals <- tolower(as.character(orig_vals))
    numeric_vals <- ifelse(grepl("recent|1", char_vals, ignore.case = TRUE), 1, 0)
  }
  # Convert to factor with explicit levels
  factor(as.character(numeric_vals), levels = c("0", "1"))
}


summary(train_processed)
summary(test_processed)

######################## Convert train/test sets into H2O data frames ###############

train_h2o <- as.h2o(train_processed)
test_h2o <- as.h2o(test_processed)

######################### Specify name of target and predictors #########################

target <- "recency_interpretation"

predictors <- setdiff(names(train_processed), target)

#################################### Random forests #########################

# H2O fits distributed random forests (DRF) 

#Define hyperparameter grid
hyper_params_drf <- list(
  ntrees = c(50, 100, 200),
  max_depth = c(10, 20, 30),
  min_rows = c(1, 5, 10)
)


# Perform grid search using H2OGrid
grid_search_drf <- h2o.grid(
  algorithm = "drf",
  grid_id = "drf_grid",
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params_drf,
  search_criteria = list(strategy = "Cartesian"),
  nfolds = 5, # change to 10 for 10-fold CV
  seed = seed
)

# Get the grid results, sorted by logloss
grid_results_drf <- h2o.getGrid(grid_id = "drf_grid", 
                                sort_by = "logloss", 
                                decreasing = FALSE)

print(grid_results_drf)

# extract best model
best_model_drf <- h2o.getModel(grid_results_drf@model_ids[[1]])

best_params_drf <- best_model_drf@allparameters
print(best_params_drf)


# Build and train the drf model based on the tuned hyperparameters:
drf <- h2o.randomForest(
  x = predictors,
  y = target,
  ntrees = best_params_drf$ntrees,
  max_depth = best_params_drf$max_depth,
  min_rows = best_params_drf$min_rows,
  #sample_rate = best_params_drf$sample_rate,
  #col_sample_rate_per_tree = best_params_drf$col_sample_rate_per_tree,
  training_frame = train_h2o,
  nfold =5,
  seed=seed,
  keep_cross_validation_predictions = TRUE ### this option is so that we can use this model in stacking
)


#

train_h2o$recency_interpretation <- h2o.asfactor(train_h2o$recency_interpretation)
test_h2o$recency_interpretation  <- h2o.asfactor(test_h2o$recency_interpretation)
# Get correct levels from training data
train_lvls <- h2o.levels(train_h2o$recency_interpretation)

test_h2o$recency_interpretation <- h2o.setLevels(
  test_h2o$recency_interpretation, 
  train_lvls
)

train_h2o <- as.h2o(train_processed)
test_h2o  <- as.h2o(test_processed)

train_h2o$recency_interpretation <- h2o.asfactor(train_h2o$recency_interpretation)
test_h2o$recency_interpretation  <- h2o.asfactor(test_h2o$recency_interpretation)

test_h2o$recency_interpretation <- h2o.setLevels(
  test_h2o$recency_interpretation,
  h2o.levels(train_h2o$recency_interpretation)
)

preds_drf_test <- h2o.predict(drf, test_h2o)

# Save predicted probabilities
preds_drf_train <- h2o.predict(drf, train_h2o)
preds_drf_test <- h2o.predict(drf, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_drf_train <- as.data.frame(preds_drf_train)
preds_drf_test <- as.data.frame(preds_drf_test)

# Check column names - H2O DRF may return "X1" for probability of class 1, or "p1"
# Use "X1" if available (DRF often uses this), otherwise try "p1" or column 3
prob_col_drf <- if("X1" %in% names(preds_drf_train)) "X1" else if("p1" %in% names(preds_drf_train)) "p1" else names(preds_drf_train)[3]

train_drf_pred <- cbind(train_processed, preds_drf_train[, prob_col_drf, drop = FALSE])
test_drf_pred <- cbind(test_processed, preds_drf_test[, prob_col_drf, drop = FALSE])

# Create confusion matrix using a threshold:

threshold <- 0.5

# training
train_drf_pred$pred_class <- factor(ifelse(train_drf_pred[[prob_col_drf]] > threshold, "1", "0"),
                                     levels = c("0", "1"))

# Ensure recency_interpretation has correct levels using helper function
train_drf_pred$recency_interpretation <- convert_recency_to_factor(train_drf_pred$recency_interpretation)

# Check if both levels exist before computing ROC
if(length(unique(na.omit(train_drf_pred$recency_interpretation))) < 2) {
  warning("Only one class present in training data. Cannot compute ROC curve.")
  print("Class distribution:")
  print(table(train_drf_pred$recency_interpretation))
} else {
  # predicted classes first then actual classes
  caret::confusionMatrix(
    train_drf_pred$pred_class,
    train_drf_pred$recency_interpretation,
    positive = "1",
    mode = "everything"
  )
  
  # actual classes first then predicted probabilities
  roc_drf_train <- roc(train_drf_pred$recency_interpretation, train_drf_pred[[prob_col_drf]],
                       levels = c("0", "1"), quiet = TRUE)
  auc(roc_drf_train)
  plot(roc_drf_train)
}

# test
test_drf_pred$pred_class <- factor(ifelse(test_drf_pred[[prob_col_drf]] > threshold, "1", "0"),
                                    levels = c("0", "1"))

# Ensure recency_interpretation has correct levels using helper function
test_drf_pred$recency_interpretation <- convert_recency_to_factor(test_drf_pred$recency_interpretation)

# Check if both levels exist before computing ROC
if(length(unique(na.omit(test_drf_pred$recency_interpretation))) < 2) {
  warning("Only one class present in test data. Cannot compute ROC curve.")
  print("Class distribution:")
  print(table(test_drf_pred$recency_interpretation))
} else {
  # predicted classes first then actual classes
  caret::confusionMatrix(
    test_drf_pred$pred_class,
    test_drf_pred$recency_interpretation,
    positive = "1",
    mode = "everything"
  )
  
  # actual classes first then predicted probabilities
  roc_drf_test <- roc(test_drf_pred$recency_interpretation, test_drf_pred[[prob_col_drf]],
                      levels = c("0", "1"), quiet = TRUE)
  auc(roc_drf_test)
  plot(roc_drf_test)
}

############################ Gradient Boosted Machines (GBM) ###################################

# Both GBM and XGboost use gradient boosting, but XGBoost enhances the process with second-order optimization, post-pruning, and explicit regularization. These additions make XGBoost more flexible and often more accurate, but also more complex and computationally intensive - hence it requires more powerful systems.

hyper_params_gbm <- list(
  ntrees = c(50, 100, 200),
  max_depth = c(3, 6, 9),
  learn_rate = c(0.01, 0.1, 0.3), # The range is 0.0 to 1.0, and the default value is 0.1.
  sample_rate = c(0.7, 1.0),
  col_sample_rate = c(0.7, 1.0),
  min_rows = c(1, 5),
  min_split_improvement = c(0, 0.1)  
)


grid_search_gbm <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid",
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params_gbm,
  nfolds = 5,
  stopping_metric = "logloss",  # change to AUC, RMSE, etc., as needed
  stopping_rounds = 3,
  seed = seed,
  search_criteria = list(strategy = "Cartesian")
)


# Get the grid results, sorted by logloss
grid_results_gbm <- h2o.getGrid(grid_id = "gbm_grid", 
                                sort_by = "logloss", 
                                decreasing = FALSE)

print(grid_results_gbm)

# extract best model
best_model_gbm <- h2o.getModel(grid_results_gbm@model_ids[[1]])

best_params_gbm <- best_model_gbm@allparameters
print(best_params_gbm)


# Build and train the gbm model based on the tuned hyperparameters:
gbm <- h2o.gbm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  ntrees = best_params_gbm$ntrees,
  max_depth = best_params_gbm$max_depth,
  learn_rate = best_params_gbm$learn_rate,
  sample_rate = best_params_gbm$sample_rate,
  col_sample_rate = best_params_gbm$col_sample_rate,
  min_rows = best_params_gbm$min_rows,
  min_split_improvement = best_params_gbm$min_split_improvement,
  seed = seed,
  nfold =5,
  keep_cross_validation_predictions = TRUE ### this option is so that we can use this model in stacking
)

# Save predicted probabilities
preds_gbm_train <- h2o.predict(gbm, train_h2o)
preds_gbm_test <- h2o.predict(gbm, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_gbm_train <- as.data.frame(preds_gbm_train)
preds_gbm_test <- as.data.frame(preds_gbm_test)

# Check column names - H2O typically returns "p1" for probability of class 1, or column named after factor level
# Use "p1" if available, otherwise use the probability column (usually column 3 or named after positive class)
prob_col_gbm <- if("p1" %in% names(preds_gbm_train)) "p1" else names(preds_gbm_train)[3]

train_gbm_pred <- cbind(train_processed, preds_gbm_train[, prob_col_gbm, drop = FALSE])
test_gbm_pred <- cbind(test_processed, preds_gbm_test[, prob_col_gbm, drop = FALSE])

# Create confusion matrix using a threshold:

threshold <- 0.5

# training
train_gbm_pred$pred_class <- factor(ifelse(train_gbm_pred[[prob_col_gbm]] > threshold, "1", "0"), 
                                     levels = c("0", "1"))

# Ensure recency_interpretation has correct levels using helper function
train_gbm_pred$recency_interpretation <- convert_recency_to_factor(train_gbm_pred$recency_interpretation)

# Check if both levels exist before computing ROC
if(length(unique(na.omit(train_gbm_pred$recency_interpretation))) < 2) {
  warning("Only one class present in GBM training data. Cannot compute ROC curve.")
  print("Class distribution:")
  print(table(train_gbm_pred$recency_interpretation))
} else {
  # predicted classes first then actual classes
  caret::confusionMatrix(
    train_gbm_pred$pred_class,
    train_gbm_pred$recency_interpretation,
    positive = "1",
    mode = "everything"
  )
  
  # actual classes first then predicted probabilities
  roc_gbm_train <- roc(train_gbm_pred$recency_interpretation, train_gbm_pred[[prob_col_gbm]],
                     levels = c("0", "1"), quiet = TRUE)
  auc(roc_gbm_train)
  plot(roc_gbm_train)
}

# test
test_gbm_pred$pred_class <- factor(ifelse(test_gbm_pred[[prob_col_gbm]] > threshold, "1", "0"),
                                    levels = c("0", "1"))

# Ensure recency_interpretation has correct levels using helper function
test_gbm_pred$recency_interpretation <- convert_recency_to_factor(test_gbm_pred$recency_interpretation)

# Check if both levels exist before computing ROC
if(length(unique(na.omit(test_gbm_pred$recency_interpretation))) < 2) {
  warning("Only one class present in GBM test data. Cannot compute ROC curve.")
  print("Class distribution:")
  print(table(test_gbm_pred$recency_interpretation))
} else {
  # predicted classes first then actual classes
  caret::confusionMatrix(
    test_gbm_pred$pred_class,
    test_gbm_pred$recency_interpretation,
    positive = "1",
    mode = "everything"
  )
  
  # actual classes first then predicted probabilities
  roc_gbm_test <- roc(test_gbm_pred$recency_interpretation, test_gbm_pred[[prob_col_gbm]],
                    levels = c("0", "1"), quiet = TRUE)
  auc(roc_gbm_test)
  plot(roc_gbm_test)
}

############################## Stacked ensemble #######################################

# Stacking is a class of algorithms that involves training a second-level “metalearner” to find the optimal combination of the base learners. Unlike bagging and boosting, the goal in stacking is to ensemble strong, diverse sets of learners together. See more info here https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html.

#Before training a stacked ensemble, you will need to train and cross-validate a set of “base models” which will make up the ensemble. In order to stack these models together, a few things are required:

# - The models must be cross-validated using the same number of folds (e.g. nfold = 5 or use the same fold_column across base learners).
# - The cross-validated predictions from all of the models must be preserved by setting keep_cross_validation_predictions = True. This is the data which is used to train the metalearner, or “combiner”, algorithm in the ensemble.

# - The models must be trained on the same training_frame. The rows must be identical, but you can use different sets of predictor columns, x, across models if you choose.


# Train a stacked ensemble using the RF and GBM from above. The metalearner_algorithm option allows you to specify a different metalearner algorithm. See https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/metalearner_algorithm.html for the options. 

stacked <- h2o.stackedEnsemble(x = predictors,
                               y = target,
                               training_frame = train_h2o,
                               #metalearner_algorithm = "gbm", # default is GLM
                               base_models = list(drf, gbm))


# Save predicted probabilities
preds_stacked_train <- h2o.predict(stacked, train_h2o)
preds_stacked_test <- h2o.predict(stacked, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_stacked_train <- as.data.frame(preds_stacked_train)
preds_stacked_test <- as.data.frame(preds_stacked_test)

# Check column names - H2O typically returns "p1" for probability of class 1, or column named after factor level
# Use "p1" if available, otherwise use the probability column (usually column 3 or named after positive class)
prob_col_stacked <- if("p1" %in% names(preds_stacked_train)) "p1" else names(preds_stacked_train)[3]

train_stacked_pred <- cbind(train_processed, preds_stacked_train[, prob_col_stacked, drop = FALSE])
test_stacked_pred <- cbind(test_processed, preds_stacked_test[, prob_col_stacked, drop = FALSE])

# Create confusion matrix using a threshold:

threshold <- 0.5

# training
train_stacked_pred$pred_class <- factor(ifelse(train_stacked_pred[[prob_col_stacked]] > threshold, "1", "0"),
                                         levels = c("0", "1"))

# Ensure recency_interpretation has correct levels using helper function
train_stacked_pred$recency_interpretation <- convert_recency_to_factor(train_stacked_pred$recency_interpretation)

# Check if both levels exist before computing ROC
if(length(unique(na.omit(train_stacked_pred$recency_interpretation))) < 2) {
  warning("Only one class present in Stacked training data. Cannot compute ROC curve.")
  print("Class distribution:")
  print(table(train_stacked_pred$recency_interpretation))
} else {
  # predicted classes first then actual classes
  caret::confusionMatrix(
    train_stacked_pred$pred_class,
    train_stacked_pred$recency_interpretation,
    positive = "1",
    mode = "everything"
  )
  
  # actual classes first then predicted probabilities
  roc_stacked_train <- roc(train_stacked_pred$recency_interpretation, train_stacked_pred[[prob_col_stacked]],
                         levels = c("0", "1"), quiet = TRUE)
  auc(roc_stacked_train)
  plot(roc_stacked_train)
}

# test
test_stacked_pred$pred_class <- factor(ifelse(test_stacked_pred[[prob_col_stacked]] > threshold, "1", "0"),
                                        levels = c("0", "1"))

# Ensure recency_interpretation has correct levels using helper function
test_stacked_pred$recency_interpretation <- convert_recency_to_factor(test_stacked_pred$recency_interpretation)

# Check if both levels exist before computing ROC
if(length(unique(na.omit(test_stacked_pred$recency_interpretation))) < 2) {
  warning("Only one class present in Stacked test data. Cannot compute ROC curve.")
  print("Class distribution:")
  print(table(test_stacked_pred$recency_interpretation))
} else {
  # predicted classes first then actual classes
  caret::confusionMatrix(
    test_stacked_pred$pred_class,
    test_stacked_pred$recency_interpretation,
    positive = "1",
    mode = "everything"
  )
  
  # actual classes first then predicted probabilities
  roc_stacked_test <- roc(test_stacked_pred$recency_interpretation, test_stacked_pred[[prob_col_stacked]],
                        levels = c("0", "1"), quiet = TRUE)
  auc(roc_stacked_test)
  plot(roc_stacked_test)
}




h2o.shutdown(prompt = FALSE)


