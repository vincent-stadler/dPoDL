import keras
from keras import backend as K
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.models import Sequential, clone_model, load_model
from keras.layers import Input, Dense, Flatten
from typing import Tuple
import numpy as np


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


def create_model(input_shape):
    """
    Initializes model
    :param input_shape: Shape of input data
    :return: compiled model
    """
    model = Sequential([
        Input(shape=input_shape),  # Explicitly define the input shape here
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])
    return model


def reset_weights(model):
    # Clone the model to create a new instance with the same architecture
    new_model = clone_model(model)
    new_model.set_weights(model.get_weights())  # Initialize with the same weights

    for layer in new_model.layers:
        # Check if the layer has weights (some layers like Dropout do not have weights)
        if isinstance(layer, keras.layers.Layer) and layer.get_weights():
            # Initialize the kernel weights
            if hasattr(layer, "kernel_initializer"):
                kernel_shape = layer.kernel.shape
                kernel = layer.kernel_initializer(kernel_shape)
            else:
                kernel = None

            # Initialize the bias weights
            if hasattr(layer, "bias_initializer") and layer.bias is not None:
                bias_shape = layer.bias.shape
                bias = layer.bias_initializer(bias_shape)
            else:
                bias = None

            # Set the weights of the layer
            if kernel is not None and bias is not None:
                layer.set_weights([kernel, bias])
            elif kernel is not None:
                layer.set_weights([kernel])

    return new_model