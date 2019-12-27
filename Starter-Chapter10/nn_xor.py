import sys
sys.path.append('../')

import numpy as np
from pyimagesearch.nn.neuralnetwork import NeuralNetwork

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

for (x, target) in zip(X, y):
    prediction = nn.predict(x)
    print("[INFO] data={}, prediction={}, truth={}".format(x, prediction, target))