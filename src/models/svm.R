##########################################################################################
#                                                                                        #
#                        H2O Package for  NNs and caret for SVM                          #
#                                                                                        #
##########################################################################################

library(h2o)
h2o.init()

###################################### Load additional packages ##############################

library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting into train and test sets
library(caret) # used for performance metric functions
library(pROC) # used for obtaining AUC



########################### Split the data into train/test sets##############################

# seed <- 123 # set a seed for reproducibility - specified earlier

set.seed(seed)

# stratified sampling is is used to maintain the proportion of class labels in your training and test sets:
split <- sample.split(train_balanced_trace$recency_interpretation, SplitRatio = 0.8) # change SplitRatio for different splits, such as change to 0.8 for 80:20 split

train_svm <- subset(train_balanced_trace, split == "TRUE")
test_svm <- subset(train_balanced_trace, split == "FALSE")

########################### Data pre-processing #############################################

# Unlike the other models we have fitted, the SVM and NN require further pre-processing on the training and test sets (separately). This is because they do not work with categorical features. In addition, all numeric variables need to be normalized so that they are on the same scale, this ensures that they all contribute equally to the training of the model.

########## Scaling of numeric features (test and training sets) ###############

# RECALL: we centre and scale the TEST set according to the scale parameters of the TRAINING set. These training set scale parameters are obtained via the preProcess function and saved as an R object:

train_norm_parameters <- preProcess(train_svm, method = c("center", "scale"))

## We then use the predict function to scale the training and test sets based on the scale parameters from the training set saved above (note the following makes no prediction, but simply scales the training and test sets to create a newly scaled set for each).

train_scaled <- predict(train_norm_parameters,train_svm)
test_scaled <- predict(train_norm_parameters,test_svm)

summary(train_scaled)
summary(test_scaled)

## NOTE: The data must be pre-processed (scaled, centered AND dummy variables)


## NOTE: The data must be pre-processed (scaled, centered AND dummy variables)

# I have to compromise the orgUnits from this training

train_scaled <- train_scaled %>%
  dplyr::select(
    -orgUnit
  )

test_scaled <- test_scaled %>%
  dplyr::select(
    -orgUnit
  )

############# Dummy variable encoding (test and training sets)  ###################

# we will use the recipes package from tidymodels:

library(recipes)

# 1. define the model so that the function knows what is the target (this defines the recipe)
rec <- recipe(recency_interpretation ~ ., data = train_scaled) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE)

# to avoid perfect linearity, one of the categories of the variable during the encoding is dropped. This speeds up the training and improves the stability of the ML model. This is done by setting one_hot = FALSE. 

# 2. Prep the recipe using the training data
rec_prep <- prep(rec, training = train_scaled)

# 3. Apply (bake) the prepped recipe on the scaled training set 
train_processed <- bake(rec_prep, new_data = NULL)

# fix names of columns which include a period:

colnames(train_processed) <- make.names(colnames(train_processed))

install.packages("janitor")
library(janitor)

# Automatically fixes everything (dots -> underscores, caps -> lower)
train_processed <- train_processed %>% 
  clean_names()

# 4. Apply (bake) the same transformations to the scaled test set
test_processed <- bake(rec_prep, new_data = test_scaled)

# fix names of columns which include a period:

colnames(test_processed) <- make.names(colnames(test_processed))

# Automatically fixes everything (dots -> underscores, caps -> lower)
test_processed <- test_processed %>% 
  clean_names()

train_processed <- as.data.frame(train_processed)
test_processed <- as.data.frame(test_processed)

summary(train_processed)
summary(test_processed)

############################## SVM #################################################

# Recall that there is at least 1 hyperparameter for the SVM, C (known as the cost or regularization parameter). This determines the penalty for misclassifying a data point, which directly affects the slack variables (measures how much each data point is allowed to violate the margin, meaning how far a data point can lie on the wrong side of the margin or even within the hyperplane itself).

# H2O does not support hyperparameter tuning for an SVM. Therefore, we will use the CARET package, specifically which allows this. Caret is already load on line 16 as we use it for the performance metrics (https://topepo.github.io/caret/train-models-by-tag.html#support-vector-machines).

# we first set up our controls for cross validation:

control <-  trainControl(method = "cv", 
                         number = 10,       # change the number of k-folders here
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary, # enable this to get sensitivity
                         savePredictions = TRUE)

# see https://topepo.github.io/caret/model-training-and-tuning.html#metrics for metrics used in the trainControl function when used for the k-fold cross-validation method, where k=10 (k can be 5 to 10) 

###################################### Linear SVM ###############################

# For a linear SVM, the only hyperparameter is C. Let's explore 3 ways of specifying values for C.

# We can specify a grid of values for the SVM to search through:

grid1 <- expand.grid(C = c(0.75, 0.9, 1)) # specify a vector of possible values

grid2 <- expand.grid(C = seq(0, 2, length = 20)) # 20 values from 0 to 20

grid3 <- expand.grid(C = seq(0, 2, by = 0.1)) # values from 0 to 2 in increments of 0.1

# Recall: C in the cost function of the SVM is a regularization parameter (referred to as the tuning parameter here) that plays a critical role in controlling the trade-off between achieving a low training error and maintaining a low model complexity for better generalization to new data. 

# a high C means giving higher penalty to the errors (slack variables). This forces the SVM to classify all training examples correctly, which can lead to overfitting if the data is noisy or not linearly separable. A high value of C such as 10000, leads to a harder margin. NOTE: Setting C = 0 would mean you're applying zero penalty for misclassification, which results in a completely unconstrained margin - so no solution. 

# a low C allows the optimizer to focus more on maximizing the margin and less on classifying all training points correctly. This can lead to a more generalized model but might increase the number of misclassifications.

# By default caret builds the SVM linear classifier using C = 1. It's possible to automatically compute SVM for different values of C and choose the optimal one that maximizes the model's cross-validation performance measure (specified by user). 


set.seed(seed)

levels(train_processed$recency_interpretation) <- make.names(levels(train_processed$recency_interpretation))

SVM_linear <- train(recency_interpretation ~.,      # state target here 
                    data = train_processed, 
                    method = "svmLinear", # for linear SVM
                    metric="Roc", # Accuracy, Kappa, Sens, Spec, ROC
                    trControl = control,
                    tuneGrid = grid1   # change between grid1, grid2 and grid3 depending on preference
)

SVM_linear  

# Plot model's performance vs different values of Cost

plot(SVM_linear)

# Print the best tuning parameter C that maximizes the model's performance

SVM_linear$bestTune

## Get predicted values for the training set

pred_train_svm_linear = predict(SVM_linear,newdata=train_processed, type="prob")

train_SVM_lin <- cbind(train_processed, pred_train_svm_linear[, "Yes", drop = FALSE]) 

### Then select a cut-off (or find optimal cut-off):

threshold <- 0.5 # specify the cut-off here (this can change to find the optimal)

# training (note the predicted probabilities are now in a column named "Yes")
train_SVM_lin$pred_class <- factor(ifelse(train_SVM_lin$Yes > threshold,"Yes","No" ))

caret::confusionMatrix(train_SVM_lin$pred_class, 
                       train_SVM_lin$income, 
                       mode="everything",
                       positive='Yes') # do not worry about the warning it gives, the class labels are coded correctly even if the reference does correspond between the actual and predicted class.

# extract ROC
# actual classes first then predicted probabilities
roc_SVM_lin_train <- roc(train_SVM_lin$income, train_SVM_lin$Yes)
auc(roc_SVM_lin_train)
plot(roc_SVM_lin_train)

## Get predicted values for the test set

pred_test_svm_linear = predict(SVM_linear,newdata=test_processed, type="prob")

test_SVM_lin <- cbind(test_processed, pred_test_svm_linear[, "X1", drop = FALSE]) 

# test (note the predicted probabilities are now in a column named "Yes")
test_SVM_lin$pred_class <- factor(ifelse(test_SVM_lin$X1 > threshold,"X1","X0" ))

caret::confusionMatrix(test_SVM_lin$pred_class, 
                       test_SVM_lin$recency_interpretation, 
                       mode="everything",
                       positive='X1')

# extract ROC
# actual classes first then predicted probabilities

#roc_SVM_lin_test <- roc(test_SVM_lin$recency_interpretation, test_SVM_lin$Yes)

roc_SVM_lin_test <- roc(
  test_SVM_lin$recency_interpretation,
  test_SVM_lin$X1
)

auc(roc_SVM_lin_test)
plot(roc_SVM_lin_test)

####################### SVM Radial ##############################

# For the Radial basis function (RBF) SVM, there is an additional hyperparameter: Gamma, (called sigma here in the train function). 
# It tells us how much each individual data point will influence the decision boundary.
# sigma is a scale parameter for the RBF kernel. It controls the width of the kernel and hence influences how the similarity between data points decreases with distance.
# A small sigma value leads to a narrower peak in the kernel function, meaning that the effect of a single training example is limited to a small neighborhood around it. This can lead to a model that fits the training data very closely, but may generalize poorly on new, unseen data (overfitting).
# A large sigma value results in a wider peak, meaning that the influence of each training example reaches further. This can cause smoother decision boundaries, potentially improving generalization but at the risk of underfitting if too large.
#In summary, low values of sigma typically produce highly non-linear decision boundaries, and large values of sigma often results in a decision boundary that is more linear. 

# Use the expand.grid to specify the search space	
grid1 <- expand.grid(sigma = c(.01, .015, 0.2),
                     C = c(0.75, 0.9, 1, 1.1, 1.25))

grid2 <- expand.grid(sigma = seq(1, 3, length = 10),
                     C = 10^6)


set.seed(seed) 

###### Note this takes a bit of time:

SVM_radial = train(recency_interpretation ~., 
                   data = train_processed,
                   method = "svmRadial", # Radial kernel
                   metric="ROC", # Accuracy, Kappa, Sens, Spec, ROC
                   trControl=control,
                   tuneGrid = grid1) # change to grid2

SVM_radial

# This SVM includes an additional hyperparameter, gamma (called sigma here). 
# It tells us how much will be the influence of the individual data points on the decision boundary.
#sigma is a scale parameter for the RBF kernel. It controls the width of the kernel and hence influences how the similarity between data points decreases with distance.
# A small sigma value leads to a narrower peak in the kernel function, meaning that the effect of a single training example is limited to a small neighborhood around it. This can lead to a model that fits the training data very closely, but may generalize poorly on new, unseen data (overfitting).
# A large sigma value results in a wider peak, meaning that the influence of each training example reaches further. This can cause smoother decision boundaries, potentially improving generalization but at the risk of underfitting if too large.
#In summary, low values of sigma typically produce highly non-linear decision boundaries, and large values of sigma often results in a decision boundary that is more linear. 

## Get predicted values for the training set

pred_train_svm_radial = predict(SVM_radial,newdata=train_processed, type="prob")

#train_SVM_radial<- cbind(train_processed, pred_train_svm_radial[, "Yes", drop = FALSE]) 
train_SVM_radial <- cbind(train_processed, pred_train_svm_radial[, "X1", drop = FALSE])

### Then select a cut-off (or find optimal cut-off):

threshold <- 0.5 # specify the cut-off here (this can change to find the optimal)

# training (note the predicted probabilities are now in a column named "Yes")
#train_SVM_radial$pred_class <- factor(ifelse(train_SVM_radial$Yes > threshold,"Yes","No" ))

train_SVM_radial$pred_class <- factor(ifelse(train_SVM_radial$X1 > threshold,"X1","X0" ))

caret::confusionMatrix(train_SVM_radial$pred_class, 
                       train_SVM_radial$recency_interpretation, 
                       mode="everything",
                       positive='X1')



# extract ROC
# actual classes first then predicted probabilities
#roc_SVM_radial_train <- roc(train_SVM_radial$recency_interpretation, train_SVM_radial$Yes)
pred_train_svm_radial = predict(SVM_radial,newdata=train_processed, type="prob")
train_SVM_radial <- cbind(train_processed, pred_train_svm_radial[, "X1", drop = FALSE])

roc_SVM_radial_train <- roc(
  train_SVM_radial$recency_interpretation,
  train_SVM_radial$X1
)
auc(roc_SVM_radial_train)
plot(roc_SVM_radial_train)

## Get predicted values for the test set

pred_test_svm_radial = predict(SVM_radial,newdata=test_processed, type="prob")

test_SVM_radial <- cbind(test_processed, pred_test_svm_radial[, "X1", drop = FALSE]) 


# test (note the predicted probabilities are now in a column named "Yes")
#test_SVM_radial$pred_class <- factor(ifelse(test_SVM_radial$Yes > threshold,"Yes","No" ))
test_SVM_radial$pred_class <- factor(ifelse(test_SVM_radial$X1 > threshold,"X1","X0" ))

caret::confusionMatrix(test_SVM_radial$pred_class, 
                       test_SVM_radial$recency_interpretation, 
                       mode="everything",
                       positive='X1')

caret::confusionMatrix(test_SVM_radial$pred_class, 
                       test_SVM_radial$recency_interpretation, 
                       mode="everything",
                       positive='X1')

# extract ROC
# actual classes first then predicted probabilities
roc_SVM_radial_test <- roc(test_SVM_radial$recency_interpretation, test_SVM_radial$X1)
auc(roc_SVM_radial_test)
plot(roc_SVM_radial_test)

############ Non-linear Polynomial SVM ####################

# Fit the model on the training set
set.seed(seed)
SVM_poly <- train(recency_interpretation ~., 
                  data = train_processed, 
                  method = "svmPoly",
                  metric="ROC", # Accuracy, Kappa, Sens, Spec, ROC
                  trControl = control # use original defined control setup - the model is automatically fitted for different values of the hyperparameters. 
)

SVM_poly

# Print the best tuning parameter (degree, scale and C). The scale parameter controls the scaling of the polynomial transformed input vectors, this adjusts how much the input data are scaled before computing the polynomial expansion. A higher scale can lead to a feature space where the distinction between different classes might be clearer, but too high a value may again risk overfitting.

SVM_poly$bestTune


## Get predicted values for the training set

pred_train_svm_poly = predict(SVM_poly,newdata=train_processed, type="prob")

train_SVM_poly <- cbind(train_processed, pred_train_svm_poly[, "X1", drop = FALSE]) 

### Then select a cut-off (or find optimal cut-off):

threshold <- 0.5 # specify the cut-off here (this can change to find the optimal)

# training (note the predicted probabilities are now in a column named "Yes")
#train_SVM_poly$pred_class <- factor(ifelse(train_SVM_poly$Yes > threshold,"Yes","No" ))
train_SVM_poly$pred_class <- factor(ifelse(train_SVM_radial$X1 > threshold,"X1","X0" ))

caret::confusionMatrix(train_SVM_poly$pred_class, 
                       train_SVM_poly$recency_interpretation, 
                       mode="everything",
                       positive='X1')

# extract ROC
# actual classes first then predicted probabilities
roc_SVM_poly_train <- roc(train_SVM_poly$recency_interpretation, train_SVM_poly$X1)
auc(roc_SVM_poly_train)
plot(roc_SVM_poly_train)

## Get predicted values for the test set

pred_test_svm_poly = predict(SVM_poly,newdata=test_processed, type="prob")

test_SVM_poly <- cbind(test_processed, pred_test_svm_poly[, "X1", drop = FALSE]) 

# test (note the predicted probabilities are now in a column named "Yes")
#test_SVM_poly$pred_class <- factor(ifelse(test_SVM_poly$Yes > threshold,"Yes","No" ))
test_SVM_poly$pred_class <- factor(ifelse(test_SVM_poly$X1 > threshold,"X1","X0" ))

caret::confusionMatrix(test_SVM_poly$pred_class, 
                       test_SVM_poly$X1, 
                       mode="everything",
                       positive='Yes')

# extract ROC
# actual classes first then predicted probabilities
roc_SVM_poly_test <- roc(test_SVM_poly$recency_interpretation, test_SVM_poly$X1)
auc(roc_SVM_poly_test)
plot(roc_SVM_poly_test)

