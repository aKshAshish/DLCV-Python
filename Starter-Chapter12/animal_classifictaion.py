import sys
sys.path.append('../')
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.shallownet import ShallowNet
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')

args = vars(ap.parse_args())

print("[INFO] loading images......")
imgPaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(X, y) = sdl.load(imgPaths, verbose=250)

X = X.astype(float) / 255.0

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Encoding labels to one hot vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Compiling the model
print("[INFO] compiling the model.......")
sgd = SGD(0.003)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(optimizer=sgd, loss="categorical_crossentropy",
              metrics=['accuracy'])

# Training the network
print("[INFO] training the network......")
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

# Evaluate the network
print("[INFO] evaluating the network........")
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(
    axis=1), target_names=['cat', 'dog', 'panda']))

# plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.xlabel("#Epochs")
plt.ylabel("Loss/Accuracy")
plt.title("Training/Validation loss and accuracy")
plt.legend()
plt.show()

