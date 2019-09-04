import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import os


from keras.models import load_model

# AlexNet Model
def AlexNet():
    # Sequential 모델 선언
    model = keras.Sequential()

    # 첫 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                  padding='SAME', activation=tf.nn.leaky_relu,
                                  input_shape=(80, 80, 3)))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME'))

    # 두 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=5, strides=1,
                                  padding='SAME', activation=tf.nn.leaky_relu))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2,
                                        padding='SAME'))

    # 세 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=3, strides=1,
                                  padding='SAME', activation=tf.nn.leaky_relu))
    # 네 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=3, strides=1,
                                  padding='SAME', activation=tf.nn.leaky_relu))
    # 다섯 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=3,
                                  padding='SAME', activation=tf.nn.leaky_relu))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2,
                                        padding='SAME'))

    # Connecting it to a Fully Connected layer
    model.add(keras.layers.Flatten())

    # 첫 번째 Fully Connected Layer
    model.add(keras.layers.Dense(4096, input_shape=(80 * 80 * 3,), activation=tf.nn.leaky_relu))
    # Add Dropout to prevent overfitting
    model.add(keras.layers.Dropout(0.4))

    # 두 번째 Fully Connected Layer
    model.add(keras.layers.Dense(2048, activation=tf.nn.leaky_relu))
    # Add Dropout
    model.add(keras.layers.Dropout(0.4))

    # 세 번째 Fully Connected Layer
    model.add(keras.layers.Dense(4, activation=tf.nn.softmax))

    return model

def AlexNet_original():
    # Sequential 모델 선언
    model = keras.Sequential()
    # 첫 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                  padding='SAME', activation='relu',
                                  input_shape=(80, 80, 3)))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME'))
    # 두 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=5, strides=1,
                                  padding='SAME', activation='relu'))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2,
                                        padding='SAME'))
    # 세 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=3, strides=1,
                                  padding='SAME', activation='relu'))
    # 네 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=3, strides=1,
                                  padding='SAME', activation='relu'))
    # 다섯 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=3,
                                  padding='SAME', activation='relu'))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2,
                                        padding='SAME'))
    # Connecting it to a Fully Connected layer
    model.add(keras.layers.Flatten())
    # 첫 번째 Fully Connected Layer
    model.add(keras.layers.Dense(4096, input_shape=(224 * 224 * 3,), activation='relu'))
    # Add Dropout to prevent overfitting
    model.add(keras.layers.Dropout(0.4))
    # 두 번째 Fully Connected Layer
    model.add(keras.layers.Dense(4096, activation='relu'))
    # Add Dropout
    model.add(keras.layers.Dropout(0.4))
    # 세 번째 Fully Connected Layer
    model.add(keras.layers.Dense(4, activation=tf.nn.softmax))
    return model


def SmallNet():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(80, 80, 3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    return model
