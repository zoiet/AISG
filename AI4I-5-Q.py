import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# Seen Data (used for training)
# Unseen Data (unavailable for training)

train = pd.read_csv("train.csv")
print(len(train))
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