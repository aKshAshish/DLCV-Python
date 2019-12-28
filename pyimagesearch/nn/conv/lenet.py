from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)

        model = Sequential([
            # 1st conv layer
            Conv2D(20, (5, 5), input_shape=input_shape,
                   activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # 2nd conv layer
            Conv2D(50, (5, 5), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # 1st FC layer
            Flatten(),
            Dense(500, activation='relu'),
            # 2nd FC layer
            Dense(classes, activation='softmax')
        ])

        return model
