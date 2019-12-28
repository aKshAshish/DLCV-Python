import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from pyimagesearch.nn.conv.lenet import LeNet
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD



# Loading Dataset
print("[INFO] loading MNIST dataset ....")
dataset = fetch_openml('mnist_784', version=1, return_X_y=True)

(X, y) = dataset
X = X.reshape(X.shape[0], 28, 28, 1)
y = y.astype(int)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Normalizing data
train_X = train_X / 255.0
test_X = test_X / 255.0

# Encoding labels to one hot vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# Compiling model
print("[INFO] compiling model .........")
opt = SGD(0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(optimizer=opt, loss="categorical_crossentropy",
              metrics=["accuracy"])

# Training Model
print("[INFO] training the network.......")
H = model.fit(train_X, train_y, batch_size=128,
              epochs=20, validation_data=(test_X, test_y))

# Evaluate the network
print("[INFO] evaluating network.......")
predictions = model.predict(test_X, batch_size=128)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(
    axis=1), target_names=[str(x) for x in lb.classes_]))

# Plotting val and acc plot
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 20), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, 20), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, 20), H.history['accuracy'], label="train_accuracy")
plt.plot(np.arange(0, 20), H.history['val_accuracy'], label="val_accuracy")
plt.title("Training/Validation loss and accuracy plot")
plt.xlabel("# Epochs")
plt.ylabel("Accuracy/Loss")
plt.legend()
plt.show()
