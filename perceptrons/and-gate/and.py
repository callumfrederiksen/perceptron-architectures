import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

training_data = np.array([
    [0, 0], # Expected 0
    [0, 1], # Expected 0
    [1, 0], # Expected 0
    [1, 1], # Expected 1
])

true_values = np.array([
    0,
    0,
    0,
    1
])

weights = np.random.rand(2)
bias = np.random.rand(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(predicted, expected):
    return 0.5 * (predicted - expected) ** 2

def dL_dwi(predicted, expected, xi):
    return (predicted - expected) * predicted * (1 - predicted) * xi

def dL_db(predicted, expected):
    return (predicted - expected) * predicted * (1 - predicted)

def feed_forward(x, w):
    return sigmoid(np.dot(x, w) + bias)

learning_rate = 0.01
epochs = 100000
loss_values = []



for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    for i, x in enumerate(training_data):
        predicted = feed_forward(x, weights)
        epoch_loss += loss(predicted, true_values[i])

        weights -= learning_rate * dL_dwi(predicted, true_values[i], x)
        bias -= learning_rate * dL_db(predicted, true_values[i])

    loss_values.append(epoch_loss)


X = [i for i in range(epochs)]
plt.plot(X, loss_values)
plt.show()

print(weights)

print("[0, 0]: ", feed_forward([0, 0], weights))
print("[0, 1]: ", feed_forward([0, 1], weights))
print("[1, 0]: ", feed_forward([1, 0], weights))
print("[1, 1]: ", feed_forward([1, 1], weights))
