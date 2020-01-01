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
import matplotlib.pyplot as plt



def step_decay(epoch):
    init_alpha = 0.01
    factor = 0.25
    drop_every = 5
    # calculate learning rate
    learning_rate = init_alpha * (factor ** np.floor((1 + epoch) / drop_every))
    return float(learning_rate)


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

# Train the network
print("[INFO] training the network.........")
cb = [LearningRateScheduler(step_decay)]
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              callbacks=cb, batch_size=128, epochs=40, verbose=1)

# Evaluate Network
print("[INFO] evaluating network............")
predictions = model.predict(test_X)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

# Plot loss accuracy curve

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, 40), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, 40), H.history['acc'], label="train_acc")
plt.plot(np.arange(0, 40), H.history['val_acc'], label="val_acc")
plt.xlabel("# Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.savefig(args['output'])
