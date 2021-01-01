# y = bo + b1*x1
# y -> Dependent variable
# x1 -> Independent variable
# bo -> constant
# b1 -> slope

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(x)
# print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Training Simple Linear Regression model on Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# plt.show()

# Visualising the Test set results

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# plt.show()

# Predicting single prediction
print(regressor.predict([[12]]))

# Coefficients of the data
print(regressor.coef_)
print(regressor.intercept_)

# Salary = 9332.95 x YearsExperience + 25609.90