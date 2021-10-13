# Author : Deepansh Dubey.
# Date   : 24/09/2021.

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
#print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-40]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-40]
diabetes_y_test = diabetes.target[-20:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)

diabetes_predicted = model.predict(diabetes_X_test)

print("Mean Squarred Error is :", mean_squared_error(diabetes_y_test, diabetes_predicted))
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_predicted)
plt.show()
