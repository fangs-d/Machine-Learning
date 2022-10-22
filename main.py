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

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-10:]

diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-10:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)

diabetes_predicted = model.predict(diabetes_X_test)

print("Mean Squared Error is :", mean_squared_error(diabetes_y_test, diabetes_predicted))
print("Root Mean Squared Error i s:", sqrt(mean_squared_error(diabetes_y_test, diabetes_predicted)))
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_predicted)
plt.show()

#Changes for Hf2022
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_new)
y_prob = clf.predict_proba(X_new)
print(y_prob)
plt.plot(X_new, y_prob[:, 1], "g-", label="virginica")
plt.show()
