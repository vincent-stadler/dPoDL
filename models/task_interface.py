from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from keras import Model


class TaskInterface(ABC):

    def __init__(self):
        self._setup()

    def _setup(self):
        pass
    

    def load_data(self):
        """
        Load dataset and split into training and test sets
        :return: tuple of (X_train, y_train), (X_test, y_test)
        """
        pass

    def create_model(self) -> Model:
        """
        Initializes and returns a compiled Keras model.

        Returns:
            Model: A compiled Keras model instance.
        """
        pass

