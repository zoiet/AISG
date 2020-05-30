##############################################
# cm = confusion_matrix(y_test, test_predictions)
# cm[<true_category_index>, <predicted_category_index>]
# print("The number of true positives is: {}".format(cm[1, 1]))
############################################## 
Accuracy = ([0,0]+[1,1]) / Total
Precision = ([1,1])/([1,1]+[0,1]) => true positive / all true value
Recall = ([1,1])/([1,1]+[1,0]) => all positive value

test_predictions = rfc.predict(X_test)
accuracy_score(y_test, test_predictions)
precision_score(y_test, test_predictions)
recall_score(y_test, test_predictions)

Overfitting models (High variance)
Underfitting models (High Bias) => failing to find the relationship between the data and the response
    + High training / testing error
    + Occurs when models are underfit
Optimal performance (Bias-Variance Trafeoff)

Error due to under/over-fitting
# Update the rfr model
# Changesing max_features can impact the training and testing error
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=11)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))

#########################################################################
# Am I underfitting?
  from sklearn.metrics import accuracy_score

test_scores, train_scores = [], []
for i in [1, 2, 3, 4, 5, 10, 20, 50]:
    rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
    rfc.fit(X_train, y_train)
    # Create predictions for the X_train and X_test datasets.
    train_predictions = rfc.predict(X_train)
    test_predictions = rfc.predict(X_test)
    # Append the accuracy score for the test and train predictions.
    train_scores.append(round(accuracy_score(y_train, train_predictions), 2))
    test_scores.append(round(accuracy_score(y_test, test_predictions), 2))
# Print the train and test scores.
print("The training scores were: {}".format(train_scores))
print("The testing scores were: {}".format(test_scores))

#############################################################################
# Holdout set validation
# Two Samples
# Create two different samples of 200 observations 
sample1 = tic_tac_toe.sample(200, random_state=1111)
sample2 = tic_tac_toe.sample(200, random_state=1171)

# Print the number of common observations 
print(len([index for index in sample1.index if index in sample2.index]))

# Print the number of observations in the Class column for both samples 
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())

############################################################################
# Cross-Validation
# Split training data into 5 sets and make one set to be the validation set everytime
# X = np.array(range(40))
# y = np.array([0] * 20 + [1] * 20)
# kf = kFold(n_splits=5)
# splits = kf.split(X)

# scikit-learn's KFold()
# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))

#####################################################################################
# Using KFold indices
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    # Fit the random forest model
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))

###########################################################
sklean's cross_val_score()
estimator: the model to Use
X: the predictor datasets
y: the response array
cv: the number of cross-validation splits

mse = make_scorer(mean_absolute_error)
cross_val_score(<estimator>, <X>, <y>, cv=5, scoring=mse)

############################################################
# scikit-learn's methods
# Instruction 1: Load the cross-validation method
from sklearn.model_selection import cross_val_score

# Instruction 2: Load the random forest regression model
from sklearn.ensemble import RandomForestRegressor

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer
from sklearn.metrics import mean_squared_error, make_scorer

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator=rfc,
                     X=X_train,
                     y=y_train,
                     cv=10,
                     scoring=mse)

# Print the mean error
print(cv.mean())

########################################################
# Leave one out cross validation (LOOCV)
  - Use when amount of training data is limited
  - Alot of computational resources

n = X.shape[0]
mse = make_scorer(mean_squared_error)
cv_results = cross_val_score(estimator, X, y, scoring=mse, cv=n)

########################################################
# Leave-one-out-cross-validation
from sklearn.metrics import mean_squared_error, make_scorer

# Create scorer
mae_scorer = make_scorer(mean_squared_error)

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(rfr, X=X, y=y, cv=y.shape[0], scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))

############################################################
Introduction to hyperparameter tuning
Hyperparameters
  - Manually set before the training Occurs

########################################################
# Creating Hyperparameters
# Review the parameters of rfr
print(rfr.get_params())

# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]

# Max features 
max_features = [4, 6, 8, 10]

########################################################
# Running a model using ranges
from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.choice(max_depth),
    min_samples_split=random.choice(min_samples_split),
    max_features=random.choice(max_features))

# Print out the parameters
print(rfr.get_params())

########################################################
# RandomizedSearchCV
Grid searching continued
- Tests every possible combination
- Additional hyperparameters increase training time exponentially
- Better methods => Random searching, Bayesian optimization

Random search 
from sklearn.model_selection import import RandomizedSearchCV

#######################################################
# Preparing for RandomizedSearch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2, 4, 6, 8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)

#######################################################
# Implementing RandomizedSearchCV
# Import the method for random search
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search =\
    RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring=scorer)

#######################################################
# Selecting your final model
rs.best_score_
rs.bes_params_
rs.best_estimators_
rs.cv_results_['params']
rs.cv_results_['mean_test_score']

#######################################################
# Selecting the best precision model
from sklearn.metrics import precision_score, make_scorer

# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(
  estimator=rfc, param_distributions=param_dist,
  scoring = precision,
  cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))