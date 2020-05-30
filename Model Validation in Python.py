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
