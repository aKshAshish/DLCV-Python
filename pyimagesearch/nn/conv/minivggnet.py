from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)

        model = Sequential([
            Conv2D(32, (3,3), padding='same', input_shape=input_shape, activation='relu'),
            BatchNormalization(),
            Conv2D(32, (3,3), padding='same', activation='relu'),
            BatchNormalization(axis=-1),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),
            Conv2D(64, (3,3), padding='same', activation='relu'),
            BatchNormalization(axis=-1),
            Conv2D(64, (3,3), padding='same', activation='relu'),
            BatchNormalization(axis=-1),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(classes, activation='softmax')
        ])

        return model