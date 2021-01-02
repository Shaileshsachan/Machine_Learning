# y = bo + b1*x1 + b2*x2 + ...... + bn*Xn
# y -> Dependent variable
# x1, x2, Xn -> Independent variable
# b1, b2, bn -> Coefficients
# bo -> constant

# Assumptions of a Linear Regression
# 1. Linearity
# 2. Homoscedasticity
# 3. Multivariate normality
# 4. Independence of Errors
# 5. Lack of multicollinearity

# Building a Model
# 1. All-in        (use all variables for prediction)
# 2. Backward Elimination     ........      P-value > Significance Level  (Remove the variables which do not affect the dependent variable or predciton)
# 3. Forward Selection        ...............     Stepwise Regression    P-value < Significance Level
# 4. Bidirectional Elimination........         Combination of Backward Elimination and Forward Selection
# 5. Score Comparison


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encode categorical data (one hot encoding)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# In Multiple Linear Regression no need of Feature Scaling

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# print(y_test)
# Training Multiple Linear Regression model on training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test results

y_pred = regressor.predict(x_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))

# Backward Elimination

import statsmodels.api as sm
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_opt = x_opt.astype(np.float64)
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_ols.summary())

x_opt = x[:, [0, 1, 2, 3, 4]]
x_opt = x_opt.astype(np.float64)
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_ols.summary())
print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))