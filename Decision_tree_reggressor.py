# used for high level data not for less or easy data with less independent variables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on whole dataset
from sklearn.tree import DecisionTreeRegressor
d_reg = DecisionTreeRegressor(random_state=0)
d_reg.fit(x, y)

# Visualizing the Decision Tree Regression results(higher resolution)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, d_reg.predict(x_grid), color='blue')
plt.xlabel('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
