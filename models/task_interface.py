from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from keras import Model


class TaskInterface(ABC):

    def __init__(self):
        self._setup()

    def _setup(self):
        pass
    

    @abstractmethod
    def load_data(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Load dataset and split into training and test sets
        :return: tuple of (X_train, y_train), (X_test, y_test)
        """
        pass

    @abstractmethod
    def create_model(self) -> Model:
        """
        Initializes and returns a compiled Keras model.

        Returns:
            Model: A compiled Keras model instance.
        """
        pass

    @abstractmethod
    def train(self):
        pass
