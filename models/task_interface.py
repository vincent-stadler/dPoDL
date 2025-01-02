from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from keras import Model


class TaskInterface(ABC):

    @abstractmethod
    def __init__(self):
        self.history = {'loss': []}
        self.batches = []
        self.lrs = []
        self.batch_size = None
        self.learning_rate = None
        self.model = None

    @abstractmethod
    def load_data(self):
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
    def train(self, **args):
        pass

    @abstractmethod
    def evaluate(self, **args):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load_model(self, **args):
        pass

