import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from typing import Tuple
import numpy as np
from keras import Model


# Load MNIST data
def load_data() -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Load dataset and split into training and test sets
    :return: tuple of (X_train, y_train), (X_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def create_model(input_shape: Tuple[int, ...]) -> Model:
    """
    Initializes and returns a compiled Keras model.

    Args:
        input_shape (Tuple[int, ...]): Shape of the input data (e.g., (28, 28) for MNIST images).

    Returns:
        Model: A compiled Keras model instance.
    """
    model = Sequential([
        Input(shape=input_shape),  # Explicitly define the input shape here
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])
    return model
