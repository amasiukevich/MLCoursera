import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from linear_regression import LinearRegression

houses = pd.read_csv('house.csv')

simple_houses_regr = houses[['sqft_living', 'price']][:100]

lin_reg = LinearRegression()

train_X, test_X, train_y, test_y = train_test_split(houses['sqft_living'], houses['price'])

train_X = np.array(train_X)[:, np.newaxis]
train_y = np.array(train_y)[:, np.newaxis]

test_X = np.array(test_X)[:, np.newaxis]
test_y = np.array(train_y)[:, np.newaxis]

lin_reg.fit(train_X, train_y)
pred = lin_reg.predict(test_X)


def compute_mse(pred, test_y):
    return (1/ len(test_y)) * np.linalg.norm(pred - test_y)

print("MSE: ", compute_mse(pred, test_y))

breakpoint()
