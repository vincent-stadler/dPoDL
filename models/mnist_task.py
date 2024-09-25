import keras
from models.task_interface import TaskInterface
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from typing import Tuple
import numpy as np
from keras import Model


class MNISTtask(TaskInterface):
    def load_data(self) -> Tuple[np.array, np.array, np.array, np.array]:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        return x_train, y_train, x_test, y_test

    def create_model(self) -> Model:
        input_shape = (28, 28)
        model = Sequential([
            Input(shape=input_shape),  # Explicitly define the input shape here
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])
        return model
