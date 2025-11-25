##########################################################################################
#                                                                                        #
#                        H2O Package for  Ensemble methods:                              #
#      Random forests, Gradient Boosted Machines (GBM), Stacked Ensembles                #  
#                                                                                        #
##########################################################################################

## NB: Only a few of the hyperparameters are considered here for tuning. This list is not exhaustive. See the H2O documentation for each algorithm for their respective hyperparameters that can be tuned. 

# currently, H2O does not have support for running the XGBoost platform in Windows, so we will not consider it at this stage. It is coming soon though. We will consider GBM instead.

library(h2o)

h2o.init()

################################# Load additional packages ##############################

library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting into train and test sets
library(caret) # used for performance metric functions
library(pROC) # used for obtaining AUC

######################################## Loading the data ####################################

# We will use the same data that was used for prac 1 and 3:

library(arules)

?AdultUCI

data("AdultUCI")

seed <- 123

# Let's use a small data set (a random sample of 1000 observations) for quick computation:
set.seed(seed)
AdultUCI <- AdultUCI[sample(nrow(AdultUCI), size = 1000), ] 

AdultUCI <- na.omit(AdultUCI) # remove observations with missing values

summary(AdultUCI)

str(AdultUCI)

# Let's drop variables that we don't need:

AdultUCI_new <- AdultUCI %>%
  dplyr::select(
    -fnlwgt,
    -relationship,
    -`capital-gain`,
    -`capital-loss`,
    -`education-num`
  )

# In addition, as we need to apply dummy variable encoding for these models, we do not want to attributes with high cardinality as that expands the dimensional. Therefore, we will drop 

# use the following to determine the number of levels (cardinality) of the factor variables:

sapply(Filter(is.factor, AdultUCI_new), nlevels)

# Drop variables with high cardinality for demonstration purposes:

AdultUCI_new <- AdultUCI_new %>%
  dplyr::select(
    -`native-country`,
    -education,
    -occupation
  )

# convert to data frame:

AdultUCI_new <- as.data.frame(AdultUCI_new)

str(AdultUCI_new)

# notice that there are ordered variables in the data set. When we want to use the H2O package, it doesn't recognize ordered columns so we must unorder these variables:

AdultUCI_new$income <- factor(AdultUCI_new$income, ordered = FALSE)
#AdultUCI_new$education <- factor(AdultUCI_new$education, ordered = FALSE) - removed

prop.table(table(AdultUCI_new$income))

# converting class labels of income target to yes/no: 

AdultUCI_new$income <- factor(ifelse(AdultUCI_new$income == "large", "Yes", "No")) #can change event of interest to small here

# Ensure "Yes" is treated as the event of interest (ref) which is required for the train function used to fit the SVM:
AdultUCI_new$income <- relevel(AdultUCI_new$income, ref = "Yes")

prop.table(table(AdultUCI_new$income))

summary(AdultUCI_new)

# drop levels for those with 0 observations:

AdultUCI_new <- droplevels(AdultUCI_new)


###################### Split the data into train/test sets##############################

# seed <- 123 # set a seed for reproducibility - specified earlier

set.seed(seed)

# stratified sampling is is used to maintain the proportion of class labels in your training and test sets:
split <- sample.split(AdultUCI_new$income, SplitRatio = 0.7) # change SplitRatio for different splits, such as change to 0.8 for 80:20 split

train <- subset(AdultUCI_new, split == "TRUE")
test <- subset(AdultUCI_new, split == "FALSE")

####################### Data pre-processing #############################################

# Unlike the other models we have fitted, the SVM and drf require further pre-processing on the training and test sets (separately). This is because they do not work with categorical features. In addition, all numeric variables need to be normalized so that they are on the same scale, this ensures that they all contribute equally to the training of the model.

########## Scaling of numeric features (test and training sets) ###############

# RECALL: we centre and scale the TEST set according to the scale parameters of the TRAINING set. These training set scale parameters are obtained via the preProcess function and saved as an R object:

train_norm_parameters <- preProcess(train, method = c("center", "scale"))

## We then use the predict function to scale the training and test sets based on the scale parameters from the training set saved above (note the following makes no prediction, but simply scales the training and test sets to create a newly scaled set for each).

train_scaled <- predict(train_norm_parameters,train)
test_scaled <- predict(train_norm_parameters,test)

summary(train_scaled)
summary(test_scaled)

## NOTE: The data must be pre-processed (scaled, centered AND dummy variables)

############# Dummy variable encoding (test and training sets)  ###################

# we will use the recipes package from tidymodels:

library(recipes)

# 1. define the model so that the function knows what is the target (this defines the recipe)
rec <- recipe(income ~ ., data = train_scaled) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE)

# to avoid perfect linearity, one of the categories of the variable during the encoding is dropped. This speeds up the training and improves the stability of the ML model. This is done by setting one_hot = FALSE. 

# 2. Prep the recipe using the training data
rec_prep <- prep(rec, training = train_scaled)

# 3. Apply (bake) the prepped recipe on the scaled training set 
train_processed <- bake(rec_prep, new_data = NULL)

# fix names of columns which include a period:

colnames(train_processed) <- make.names(colnames(train_processed))

# 4. Apply (bake) the same transformations to the scaled test set
test_processed <- bake(rec_prep, new_data = test_scaled)

# fix names of columns which include a period:

colnames(test_processed) <- make.names(colnames(test_processed))

summary(train_processed)
summary(test_processed)

######################## Convert train/test sets into H2O data frames ###############

train_h2o <- as.h2o(train_processed)
test_h2o <- as.h2o(test_processed)

######################### Specify name of target and predictors #########################

target <- "income"

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


# Save predicted probabilities
preds_drf_train <- h2o.predict(drf, train_h2o)
preds_drf_test <- h2o.predict(drf, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_drf_train <- as.data.frame(preds_drf_train)
preds_drf_test <- as.data.frame(preds_drf_test)

train_drf_pred <- cbind(train_processed, preds_drf_train[, "Yes", drop = FALSE]) # drop = FALSE keeps the original name of this column (yes) in the resulting data frame
test_drf_pred <- cbind(test_processed, preds_drf_test[, "Yes", drop = FALSE])

# Create confusion matrix using a threshold:

threshold <- 0.5

# training
train_drf_pred$pred_class <- factor(ifelse(train_drf_pred$Yes > threshold,"Yes","No"))

# predicted classes first then actual classes
caret::confusionMatrix(
  train_drf_pred$pred_class,
  train_drf_pred$income,
  positive = "Yes",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_drf_train <- roc(train_drf_pred$income, train_drf_pred$Yes)
auc(roc_drf_train)
plot(roc_drf_train)

# test
test_drf_pred$pred_class <- factor(ifelse(test_drf_pred$Yes > threshold, "Yes", "No"))

# predicted classes first then actual classes
caret::confusionMatrix(
  test_drf_pred$pred_class,
  test_drf_pred$income,
  positive = "Yes",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_drf_test <- roc(test_drf_pred$income, test_drf_pred$Yes)
auc(roc_drf_test)
plot(roc_drf_test)

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

train_gbm_pred <- cbind(train_processed, preds_gbm_train[, "Yes", drop = FALSE]) # drop = FALSE keeps the original name of this column (yes) in the resulting data frame
test_gbm_pred <- cbind(test_processed, preds_gbm_test[, "Yes", drop = FALSE])

# Create confusion matrix using a threshold:

threshold <- 0.5

# training
train_gbm_pred$pred_class <- factor(ifelse(train_gbm_pred$Yes > threshold,"Yes","No"))

# predicted classes first then actual classes
caret::confusionMatrix(
  train_gbm_pred$pred_class,
  train_gbm_pred$income,
  positive = "Yes",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_gbm_train <- roc(train_gbm_pred$income, train_gbm_pred$Yes)
auc(roc_gbm_train)
plot(roc_gbm_train)

# test
test_gbm_pred$pred_class <- factor(ifelse(test_gbm_pred$Yes > threshold, "Yes", "No"))

# predicted classes first then actual classes
caret::confusionMatrix(
  test_gbm_pred$pred_class,
  test_gbm_pred$income,
  positive = "Yes",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_gbm_test <- roc(test_gbm_pred$income, test_gbm_pred$Yes)
auc(roc_gbm_test)
plot(roc_gbm_test)

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

train_stacked_pred <- cbind(train_processed, preds_stacked_train[, "Yes", drop = FALSE]) # drop = FALSE keeps the original name of this column (yes) in the resulting data frame
test_stacked_pred <- cbind(test_processed, preds_stacked_test[, "Yes", drop = FALSE])

# Create confusion matrix using a threshold:

threshold <- 0.5

# training
train_stacked_pred$pred_class <- factor(ifelse(train_stacked_pred$Yes > threshold,"Yes","No"))

# predicted classes first then actual classes
caret::confusionMatrix(
  train_stacked_pred$pred_class,
  train_stacked_pred$income,
  positive = "Yes",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_stacked_train <- roc(train_stacked_pred$income, train_stacked_pred$Yes)
auc(roc_stacked_train)
plot(roc_stacked_train)

# test
test_stacked_pred$pred_class <- factor(ifelse(test_stacked_pred$Yes > threshold, "Yes", "No"))

# predicted classes first then actual classes
caret::confusionMatrix(
  test_stacked_pred$pred_class,
  test_stacked_pred$income,
  positive = "Yes",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_stacked_test <- roc(test_stacked_pred$income, test_stacked_pred$Yes)
auc(roc_stacked_test)
plot(roc_stacked_test)




h2o.shutdown(prompt = FALSE)


