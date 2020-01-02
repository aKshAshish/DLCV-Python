import sys
sys.path.append('../')
import argparse
import numpy as np
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
import matplotlib.pyplot as plt
import os

# INFO of PID
print("[INFO] process id {}".format(os.getpgid()))

# Comand line arguments parser
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='Output path')

args = vars(ap.parse_args())

# Label names
label_names = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]

# Loading data
print("[INFO] loading data........")
((train_X, train_y), (test_X, test_y)) = cifar10.load_data()

# Normalize data
train_X = train_X.astype(float) / 255.0
test_X = test_X.astype(float) / 255.0

# One hot encoding for labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Compile model
print("[INFO] compiling model..........")
model = MiniVGGNet.build(32, 32, 3, len(label_names))
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss="categorical_crossentropy",
              metrics=['accuracy'])

# construct set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])

cb = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# Train the network
print("[INFO] training the network.........")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              callbacks=cb, batch_size=64, epochs=40, verbose=1)