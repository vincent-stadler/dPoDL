import keras
from dPoDL.models.task_interface import TaskInterface
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from typing import Tuple
import numpy as np
from keras import Model
from keras.optimizers import Adam



class MNISTtask(TaskInterface):

    def __init__(self, save_path):
        super().__init__()
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.batch_size = None
        self.learning_rate = None
        self.save_path = save_path


    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train.astype("float32") / 255.0
        self.x_test = x_test.astype("float32") / 255.0
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)


    def evaluate(self):
        return self.model.evaluate(self.x_train, self.y_train, verbose=0)

    def train(self, max_epoch, callbacks):
        self.model.fit(self.x_train, self.y_train, epochs=max_epoch, batch_size=self.batch_size, verbose=0,
                  callbacks=callbacks)

    def create_model(self):
        input_shape = (28, 28)
        model = Sequential([
            Input(shape=input_shape),  # Explicitly define the input shape here
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])
        model.compile(optimizer=Adam(self.learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model

    def save(self):
        self.model.save(self.save_path)

