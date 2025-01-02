import keras
from dPoDL.models.task_interface import TaskInterface
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from typing import Tuple
import numpy as np
from keras import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt


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
        self.batches = [8 * i for i in range(1, 17)]
        self.lrs = [0.001 * i for i in range(1, 11)]
        self.history = {}

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        (x_train, y_train) = (x_train[:200], y_train[:200])
        self.x_train = x_train.astype("float32") / 255.0
        self.x_test = x_test.astype("float32") / 255.0
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)
        print(f"Loaded MNIST dataset\nShape X: {self.x_train.shape}\nShape Y: {self.y_train.shape}")

    def evaluate(self, test_data=True):
        results = self.model.evaluate(self.x_test, self.y_test, verbose=0) if test_data else self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return results

    def train(self, epochs, callbacks=None):
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 epochs=epochs,
                                 batch_size=self.batch_size,
                                 verbose=0,
                                 callbacks=callbacks,
                                 validation_data=(self.x_test, self.y_test))
        for key in history.history:
            if key not in self.history:
                self.history[key] = []
            self.history[key].extend(history.history[key])

    def plot_metrics(self):
        # Plot accuracy
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history:
            plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()

        # Save the plot to a file
        img_path = f"MNIST-lr{self.learning_rate}-bs{self.batch_size}.png"
        plt.savefig(img_path)

        # Display the plot
        plt.show()

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

    def load_model(self, load_path):
        self.model = keras.models.load_model(load_path)
