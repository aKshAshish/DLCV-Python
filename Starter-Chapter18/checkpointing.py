import sys
sys.path.append('../')
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np



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
optimizer = SGD(learning_rate=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(
    "./model.hdf5", monitor="val_loss", mode='min', save_best_only=True, verbose=1)
cb = [checkpoint]
# Training model
print("[INFO] training network........")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              epochs=40, batch_size=64, verbose=1, callbacks=cb)

# Evaluating network
print("[INFO] evaluating network...........")
predictions = model.predict(test_X, batch_size=64)
print(classification_report(test_y.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=label_names))

# Plot loss and acc
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, 40), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, 40), H.history['accuracy'], label="train_accuracy")
plt.plot(np.arange(0, 40), H.history['val_accuracy'], label="val_accuracy")
plt.title("Training/Validation loss and accuracy plot")
plt.xlabel("# Epochs")
plt.ylabel("Accuracy/Loss")
plt.legend()
plt.show()
