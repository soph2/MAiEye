import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

model = tf.keras.models.load_model('./Janghoo_model.h5')
