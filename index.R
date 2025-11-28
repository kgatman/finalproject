##########################################################################################
#                                  Makhate Makhate                                       #
#                                  PGDip - Data Science                                  #
#                                  Westville - UKZN                                      #
##########################################################################################

##########################################################################################
#                                  File Structure                                        #
#                                                                                        #
#              project/                                                                  #
#              |   index.R          # this file does nothing really it's just a mind map #                                                #
#              |   data/                                                                 #                                                                   #
#              |   src/                                                                  #
#              |   |   |-- pre.R.        # for data cleaning                             #
#              |   |   |-- eda.R.        # for exploratory data analysis                 #
#              |   |   |-- split.R       # for splitting the data                        #
#              |   |   |-- models/       # for the models                                #
#              |-- plots/                                                                #
#              |   |-- all plots have been extracted to this folder for later reuse.     #
#                                                                                        #
##########################################################################################

##########################################################################################
#                                  Data Preprocessing                                    #
# Before splitting the data, we're going to do the following                             #
# ----- Cleaning                                                                         #
# ----- Removing invalid rows or rows with null values                                   #
# ----- Removing highly correlated variables to about multicollinearity                  #
##########################################################################################

# ----> src/pre.R

##########################################################################################
#                                  EDA                                                   #
#                                                                                        #
#              A closer look the predictor and target variables                          #
#                                                                                        #
##########################################################################################

# ----> src/eda_.R

##########################################################################################
#                                  Splitting the data                                    #
# To avoid data leakage, i've decided to split the data on 80:20 ratio                   #
#                                                                                        #
##########################################################################################

# ----> src/split.R

##########################################################################################
#                                  Balancing.                                            #
# For balancing, I'm smitten by the SMOTE balancing to oversample the Long-term          #
# class in the Recency_Interpretation variable.                                          #
# But after careful consideration, i'm going with a mixture of under sampling and.       #
# and over-sampling because SMOTE seems to cause overfitting                             #
##########################################################################################

# ----> src/blnc.R

##########################################################################################
#                                 Modeling                                               #
# The following models will be run on the data:                                          #
# Evaluations will follow each and every model                                                                                       #
##########################################################################################

# ----> src/models/nb.R          # Naive Bayes Model
# ----> src/models/logistic.R    # Logistic Regression Model
# ----> src/models/svm.R         # Support Vector Machines Model
#                  |---Support Vector Machines
#                  |---Neural Network
# ----> src/models/ensemble.R    # Ensemble Methods
#                  |---Random Forests
#                  |---XGboost
#                  |---Gradient Boosted Machine
