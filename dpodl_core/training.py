# From dpodl.py to training.py
from keras.optimizers import Adam
import random
import numpy as np
from models.model_utils import reset_weights
from models.mnist_task import create_model
from callbacks.callbacks import *

BATCHES = [8 * i for i in range(1, 17)]
LRS = [0.001 * i for i in range(1, 11)]


def hash_to_architecture(hash_val, model):
    bsize = int(hash_val, 16) % len(BATCHES)
    lr = int(hash_val, 16) % len(LRS)

    # Generate initial learnable parameters by setting a seed
    seed = int(hash_val, 16) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

    # Reinitialize the model weights
    new_model = reset_weights(model)

    return BATCHES[bsize], LRS[lr], new_model


def _training(x_train, y_train, model, batch_size, threshold, max_epoch, max_iteration, save_path):
    # Create an instance of the custom batch logger callback
    batch_logger = BatchLogger()
    # Create an instance of the early stopping callback
    early_stopping = EarlyStoppingByAccuracy(threshold=threshold, verbose=0)

    iteration = 0
    accuracy = 0
    while accuracy < threshold and iteration < max_iteration:
        # print(f"Iteration {iteration + 1}/{max_iteration}")
        # Train the model with the custom callbacks
        model.fit(x_train, y_train, epochs=max_epoch, batch_size=batch_size, verbose=0,
                  callbacks=[batch_logger, early_stopping])
        # Evaluate accuracy on the training set
        _, accuracy = model.evaluate(x_train, y_train, verbose=0)
        # print(f"Current Training Accuracy = {accuracy:.4f}")
        iteration += 1

    # if accuracy >= threshold:
    #     print(f"Training complete: Model achieved accuracy {accuracy:.4f} which meets or exceeds the threshold of {threshold}.")
    # else:
    #     print(f"Training stopped: Reached max iterations ({max_iteration}) with final accuracy {accuracy:.4f}.")

    # Save the trained model using the native Keras format
    model.save(save_path)

    return model


def _training_fresh(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path):
    init_model = create_model(x_train.shape[1:])
    batch_size, learning_rate, model = hash_to_architecture(hash_val, init_model)
    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    return _training(x_train, y_train, model, batch_size, threshold, max_epoch, max_iteration, save_path)


def main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path, load_path=None,
                  referred=None):
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

