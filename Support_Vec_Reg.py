import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
y = y.reshape(len(y), 1)      # Creating a 2-D array

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Training SVR model to whole dataset

from sklearn.svm import SVR
regresssor = SVR(kernel='rbf')
regresssor.fit(x, y)

# Predicting a new result
print(sc_y.inverse_transform(regresssor.predict(sc_x.transform([[6.5]]))))

# Visualizing the SVR result
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regresssor.predict(x)), color='blue')
plt.title('Prediciting salary`')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

# Visualizing the SVR results(for higher resolution and smoother curve)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regresssor.predict(sc_x.transform(x_grid))), color='blue')
plt.title('Truth or Bluff')
plt.xlabel("Position level")
plt.ylabel('Salary')
plt.show()