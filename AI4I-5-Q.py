import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import import RandomizedSearchCXV

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor


# Seen Data (used for training)
# Unseen Data (unavailable for training)

train = pd.read_csv("train.csv")
# print(len(train))
print(train.info())
print('Median value for SalePrice: '+ str(np.median(train['SalePrice'])))
print('Unique Values in SaleType: '+ str(len(train.SaleType.unique())))

# plt.hist(train.SalePrice, bins = 30)
# plt.show()

print('Correlation between 1stFlrSF and SalePrice: ' + str(train['1stFlrSF'].corr(train['SalePrice'])))
print('Correlation between YearBuilt and SalePrice: ' + str(train['YearBuilt'].corr(train['SalePrice'])))
print('Correlation between OverallQual and SalePrice: ' + str(train['OverallQual'].corr(train['SalePrice'])))
print('Correlation between BsmtFinSF1 and SalePrice: ' + str(train['BsmtFinSF1'].corr(train['SalePrice'])))

print('Missing value for PoolQC : '+ str((train.PoolQC.isnull().sum())))
print('Missing value for MiscFeature  : '+ str((train.MiscFeature .isnull().sum())))
print('Missing value for LotFrontage : '+ str((train.LotFrontage.isnull().sum())))


# Create arrays for features and target variable
y = train['life'].values
X = train['fertility'].values

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

##################################
# scikit-learn dataset is called bunch
# iris.data.shape => samples are in rows, features are in coluumn


# X = pd.get_dummies(train.iloc[:,0:9])
# y = train.iloc[:, 9]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

# Mean absolute error (MAE)
# rfr = RandomForestRegressor(n_estimators=500, random_state=1111)
# rfr.fit(X_train, y_train)
# test_predictions = rfr.predict(X_test)
# sum(abs(y_test - test_predictions))/len(test_predictions)
# OR
# mean_absolute_error(y_test, test_predictions)

# Mean squared error (MSE)
# sum(abs(y_test - test_predictions)**2)/len(test_predictions)
# OR
# mean_squared_error(y_test, test_predictions)

# # Manually calculate the MAE
# n = len(predictions)
# mae_one = sum(abs(y_test - predictions)) / n
# print('With a manual calculation, the error is {}'.format(mae_one))

# # Use scikit-learn to calculate the MAE
# mae_two = mean_absolute_error(y_test, predictions)
# print('Using scikit-lean, the error is {}'.format(mae_two))


