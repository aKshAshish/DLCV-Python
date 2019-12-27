import sys
sys.path.append('../')
import numpy as np
from pyimagesearch.nn.perceptron import Perceptron

#OR dataset

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

#Defining perceptron
print("[INFO] defining perceptron an training it")
perceptron = Perceptron(X.shape[1], alpha=0.1)
perceptron.fit(X, y)

#Testing the perceptron
print("[INFO] testing the perceptron")

for (x, target) in zip(X, y):
    prediction = perceptron.predict(x)
    print("[INFO] data={}, ground truth={}, prediction={}". format(x, target, prediction))