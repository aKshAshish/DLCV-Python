from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.core import Flatten

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        # Building the model
        model = Sequential([
            Conv2D(32, (3,3), padding='same', input_shape=inputShape, activation='relu'),
            Flatten(),
            Dense(classes, activation='softmax')
        ])

        return model