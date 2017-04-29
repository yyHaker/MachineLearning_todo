import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
"""
最小二乘法的线性回归，这里使用了多个特征数据
=========================================================
Linear Regression Example
=========================================================

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print(diabetes)
# Use multiple feature
diabetes_X = diabetes.data[:, 0: 5]
print(diabetes_X)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error(均方误差)
print("Mean squared error: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_y_test, regr.predict(diabetes_X_test))
plt.plot(diabetes_y_test, diabetes_y_test)

plt.xlabel('true value')
plt.ylabel('predict value')
# plt.xticks(())
# plt.yticks(())

plt.show()
