import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
from pyimagesearch.nn.conv.shallownet import ShallowNet



# Loading the data
print("[INFO] loading cifar-10 data.........")
((train_X, train_y), (test_X, test_y)) = cifar10.load_data()
train_X = train_X.astype(float) / 255.0
test_X = test_X.astype(float) / 255.0

# One hot encode the labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Label names
label_names = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]

# Compile the model
print("[INFO] compiling the model......")
optimizer = SGD(0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

# Training model
print("[INFO] training network........")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              epochs=50, batch_size=32, verbose=1)

# Evaluating network
print("[INFO] evaluating network...........")
predictions = model.predict(test_X, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=label_names))

# Plot loss and acc
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 50), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, 50), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, 50), H.history['accuracy'], label="train_accuracy")
plt.plot(np.arange(0, 50), H.history['val_accuracy'], label="val_accuracy")
plt.title("Training/Validation loss and accuracy plot")
plt.xlabel("#Epochs")
plt.ylabel("Accuracy/Loss")
plt.legend()
plt.show()

