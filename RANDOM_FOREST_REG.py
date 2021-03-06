import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data1.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regresor = RandomForestRegressor(n_estimators=100, random_state=0)
regresor.fit(x_train, y_train)

y_pred = regresor.predict(x_test)
np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))