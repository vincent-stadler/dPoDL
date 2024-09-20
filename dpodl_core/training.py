from keras.optimizers import Adam
import random
import numpy as np
from keras.models import Model
from models.model_utils import reset_weights, load_model
from models.mnist_task import create_model
from callbacks.callbacks import BatchLogger, EarlyStoppingByAccuracy
from typing import Optional, Tuple

BATCHES = [8 * i for i in range(1, 17)]
LRS = [0.001 * i for i in range(1, 11)]


def hash_to_architecture(hash_val: str, model: Model) -> Tuple[int, float, Model]:
    """
    Generates a batch size, learning rate, and reinitializes the model based on the provided hash value.

    Args:
        hash_val (str): A hexadecimal string hash used to generate model architecture and parameters.
        model (Model): A Keras model whose weights will be reset.

    Returns:
        Tuple[int, float, Model]: The batch size, learning rate, and the reinitialized model.
    """
    bsize = int(hash_val, 16) % len(BATCHES)
    lr = int(hash_val, 16) % len(LRS)

    # Generate initial learnable parameters by setting a seed
    seed = int(hash_val, 16) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

    # Reinitialize the model weights
    new_model = reset_weights(model)

    return BATCHES[bsize], LRS[lr], new_model


def _training(x_train: np.ndarray, y_train: np.ndarray, model: Model, batch_size: int, threshold: float,
              max_epoch: int, max_iteration: int, save_path: str) -> Model:
    """
    Trains the model until the desired accuracy threshold is reached or the maximum number of iterations is met.

    Args:
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        model (Model): The Keras model to train.
        batch_size (int): Batch size for training.
        threshold (float): Desired accuracy threshold for early stopping.
        max_epoch (int): Maximum number of epochs per training iteration.
        max_iteration (int): Maximum number of iterations to attempt.
        save_path (str): Path to save the final trained model.

    Returns:
        Model: The trained Keras model.
    """
    batch_logger = BatchLogger()  # Custom batch logger callback
    early_stopping = EarlyStoppingByAccuracy(threshold=threshold, verbose=0)  # Early stopping callback

    iteration = 0
    accuracy = 0
    while accuracy < threshold and iteration < max_iteration:
        print(f"Iteration {iteration + 1}/{max_iteration}")
        model.fit(x_train, y_train, epochs=max_epoch, batch_size=batch_size, verbose=0,
                  callbacks=[batch_logger, early_stopping])
        # Evaluate accuracy on the training set
        _, accuracy = model.evaluate(x_train, y_train, verbose=0)
        print(f"Current Training Accuracy = {accuracy:.4f}")
        iteration += 1

    if accuracy >= threshold:
        print(f"Training complete: Model achieved accuracy {accuracy:.4f} which meets or exceeds the threshold of {threshold}.")
    else:
        print(f"Training stopped: Reached max iterations ({max_iteration}) with final accuracy {accuracy:.4f}.")

    model.save(save_path)  # Save the trained model
    return model


def _training_fresh(hash_val: str, x_train: np.ndarray, y_train: np.ndarray, threshold: float, max_epoch: int,
                    max_iteration: int, save_path: str) -> Model:
    """
    Trains a new model initialized from scratch based on a hash value.

    Args:
        hash_val (str): Hash value used to generate batch size and learning rate for the model.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        threshold (float): Desired accuracy threshold for early stopping.
        max_epoch (int): Maximum number of epochs per training iteration.
        max_iteration (int): Maximum number of iterations to attempt.
        save_path (str): Path to save the final trained model.

    Returns:
        Model: The trained Keras model.
    """
    init_model = create_model(x_train.shape[1:])
    batch_size, learning_rate, model = hash_to_architecture(hash_val, init_model)
    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    return _training(x_train, y_train, model, batch_size, threshold, max_epoch, max_iteration, save_path)


def main_training(hash_val: str, x_train: np.ndarray, y_train: np.ndarray, threshold: float, max_epoch: int,
                  max_iteration: int, save_path: str, load_path: Optional[str] = None,
                  referred: Optional[Model] = None) -> Model:
    """
    Main function to handle model training. It either trains a fresh model or continues training from a preloaded one.

    Args:
        hash_val (str): Hash value to generate batch size and learning rate.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        threshold (float): Desired accuracy threshold for early stopping.
        max_epoch (int): Maximum number of epochs per training iteration.
        max_iteration (int): Maximum number of iterations to attempt.
        save_path (str): Path to save the trained model.
        load_path (Optional[str], optional): Path to load a pre-trained model. Defaults to None.
        referred (Optional[Model], optional): Pre-trained Keras model for further training. Defaults to None.

    Returns:
        Model: The trained or fine-tuned Keras model.
    """
    if load_path is None and referred is None:
        return _training_fresh(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path)
    elif load_path is not None and referred is None:
        # Load the pre-trained model from the given path
        referred = load_model(load_path)

    # Generate batch size, learning rate from the hash value
    batch_size, learning_rate, _ = hash_to_architecture(hash_val, referred)
    # Set the learning rate for the optimizer without resetting the learnable parameters
    referred.optimizer.learning_rate.assign(learning_rate)

    return _training(x_train, y_train, referred, batch_size, threshold, max_epoch, max_iteration, save_path)
