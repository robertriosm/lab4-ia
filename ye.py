
import numpy as np
from matplotlib import pyplot as plt

from gradient_descent import gradient_descent
from linear_cost import linear_cost, linear_cost_gradient
from pandas import read_csv, to_datetime

data = read_csv('kc_house_data.csv', header=0, index_col=0)
# print(data)
data = data.drop(columns=['date'])

# data['date'] = data['date'].apply(to_datetime).values.astype(float)
# print(data)

X = np.array(data)
X = np.nan_to_num(X).T
print(X[:,4])
y = X[:,4].reshape(X.shape[0],1)

print("y shape",y.shape)
print("X shape",X.shape)

# TRAINING_SET_SIZE = 200

# x = np.linspace(-10, 30, TRAINING_SET_SIZE)

# X = np.vstack(
#     (
#         np.ones(TRAINING_SET_SIZE),
#         x,
#         x ** 2,
#         x ** 3,
#     )
# ).T

# y = (5 + 2 * x ** 3 + np.random.randint(-9000, 9000, TRAINING_SET_SIZE)).reshape(
#     TRAINING_SET_SIZE,
#     1
# )


m, n = X.shape
theta_0 = np.random.rand(n, 1)
r_theta, costs, thetas = gradient_descent(
    theta_0=theta_0,
    cost_function=linear_cost,
    cost_function_gradient=linear_cost_gradient,
    learning_rate=0.00000001,
    threshold=0.001,
    max_iter=10000,
    params=(X,y,1),
)


for test_theta in thetas:
    pass

plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], X @ test_theta, color='green')
plt.show()

print("test_theta", test_theta)