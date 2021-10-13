#Author : Deepansh Dubey.
#Date   : 10/10/2021.
#Logistic Regression classifier to predict iris virginica.

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(iris.keys())
print(iris['data'])
print(iris['target'])
print(iris['DESCR'])

#model fitting
X = iris["data"][:, 3:]
Y = (iris["target"] == 2).astype(np.int32)

#model training
clf = LogisticRegression()
clf.fit(X, Y)
pred = clf.predict(([[1.3232]]))
print(pred)

#using matplotlib to visualise

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_new)
y_prob = clf.predict_proba(X_new)
print(y_prob)
plt.plot(X_new, y_prob[:, 1], "g-", label="virginica")
plt.show()
