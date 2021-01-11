import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import apyori

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
# print(dataset.head())
transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i, j]) for j in range(20)])
# print(transactions)

# Training the Apriori model on dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length = 2, max_length = 2)

results = list(rules)
# print(results)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidence, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Left Hand side', 'Right Hand side', 'Supports', 'Confidence', 'Lifts'])
# print(resultsinDataFrame)

print(resultsinDataFrame.nlargest(n = 10, columns='Lifts'))
# print(resultsinDataFrame)