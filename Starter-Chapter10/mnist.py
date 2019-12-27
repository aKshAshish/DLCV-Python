from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="Output to loss accuracy plot")
args = vars(ap.parse_args())


# Loading Dataset
print("[INFO] loading MNIST dataset ....")
dataset = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

(X, y) = dataset

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Normalizing data
train_X = train_X / 255.0
test_X = test_X / 255.0

# Encoding labels to one hot vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# Defining model
model = Sequential([
    Dense(256, input_shape=(784, ), activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
# Training network using SGD
sgd = SGD(0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
H = model.fit(train_X, train_y, validation_data=(
    test_X, test_y), epochs=100, batch_size=128)

# Evaluating network
print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=128)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# print(H.history.keys())
#Plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='validation-loss')
plt.plot(np.arange(0, 100), H.history['accuracy'], label='train_accuracy')
plt.plot(np.arange(0, 100), H.history['val_accuracy'], label='validation_accuracy')
plt.title("Training Loss and accuracy")
plt.xlabel("#Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])