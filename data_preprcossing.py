import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
# print(x)

# encoding categorical data using one hot encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)

# encoding the dependent variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# Splitting the dataset into the Training and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

# Feature Scaling
# Standard Deviation = {sqrt {    {sum (x - mean(x))**2} / {N(Size of the population)}   } }

# Standardisation = x(stand) = x - mean(x) / standard deviation(x)   .....Value between +3 and -3
# Normalisation = x-min(x) / max(x)-min(x)     ......Value between 0 and 1

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

# print(x_train)
# print(x_test)


