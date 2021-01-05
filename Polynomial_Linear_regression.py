# y = bo + b1X1 + b2X1^2 + ..... + bnX1^n

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# No splittiing of data into train and test for leveraging maximum data

# Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Polynomial Regression(Higher degree and smoother curve)
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=4)
x_poly = pol_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg2.predict(x_poly), color='blue')
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

# Predicting a new result
print(lin_reg.predict([[6.5]]))     # Predcition way higher because of Linear Regression
print(lin_reg2.predict(pol_reg.fit_transform([[6.5]])))
