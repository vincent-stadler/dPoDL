import hashlib
import random
import keras
from keras import backend as K
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.models import Sequential, clone_model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
import numpy as np

from DL_task import create_model, load_data

BATCHES = [8 * i for i in range(1, 17)]
LRS = [0.001 * i for i in range(1, 11)]

class BatchLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        # print(f"Batch {batch}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")
        pass

class EarlyStoppingByBatchAccuracy(Callback):
    def __init__(self, threshold, monitor="accuracy", verbose=0):
        super(EarlyStoppingByBatchAccuracy, self).__init__()
        self.threshold = threshold
        self.monitor = monitor
        self.verbose = verbose

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        accuracy = logs.get(self.monitor)
        if accuracy is not None and accuracy >= self.threshold:
            self.model.stop_training = True
            if self.verbose > 0:
                print(f"\nBatch {batch + 1}: early stopping because {self.monitor} reached {accuracy:.4f}")

class EarlyStoppingByAccuracy(Callback):
    def __init__(self, threshold, monitor="accuracy", verbose=0):
        super(EarlyStoppingByAccuracy, self).__init__()
        self.threshold = threshold
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get(self.monitor)
        if accuracy is not None and accuracy >= self.threshold:
            self.model.stop_training = True
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: early stopping because {self.monitor} reached {accuracy:.4f}")


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

# Hash to architecture: Generate batch size, learning rate, and reinitialize model weights
def hash_to_architecture(hash_val, model):
    bsize = int(hash_val, 16) % len(BATCHES)
    lr = int(hash_val, 16) % len(LRS)
    
    # Generate initial learnable parameters by setting a seed
    seed = int(hash_val, 16) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    
    # Reinitialize the model weights
    new_model = reset_weights(model)
    
    return BATCHES[bsize], LRS[lr], new_model

# Proof-of-Work: Generate a valid nonce and new hash based on difficulty
def pre_pow(prev_hash, difficulty):
    nonce = 0
    while True:
        combined = f"{prev_hash}{nonce}".encode()
        new_hash = hashlib.sha256(combined).hexdigest()
        new_hash_bin = bin(int(new_hash, 16))[2:].zfill(256) #zfill ensures the binary string always represents 256 bits
        if new_hash_bin[:difficulty] == "0" * difficulty:  # checking if satisfies given threshold
            return nonce, new_hash_bin
        nonce += 1

def _training_(x_train, y_train, model, batch_size, threshold, max_epoch, max_iteration, save_path):
    # Create an instance of the custom batch logger callback
    batch_logger = BatchLogger()
    # Create an instance of the early stopping callback
    early_stopping = EarlyStoppingByAccuracy(threshold=threshold, verbose=0)

    iteration = 0
    accuracy = 0
    while accuracy < threshold and iteration < max_iteration:
        # print(f"Iteration {iteration + 1}/{max_iteration}")
        # Train the model with the custom callbacks
        model.fit(x_train, y_train, epochs=max_epoch, batch_size=batch_size, verbose=0, callbacks=[batch_logger, early_stopping])
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

def _training_fresh_(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path):
    init_model = create_model(x_train.shape[1:])
    batch_size, learning_rate, model = hash_to_architecture(hash_val, init_model)
    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    return _training_(x_train, y_train, model, batch_size, threshold, max_epoch, max_iteration, save_path)

def main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path, load_path=None, referred=None):
    if load_path is None and referred is None:
        return _training_fresh_(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path)
    elif load_path is not None and referred is None:
        # Load the pre-trained model from the given path
        referred = load_model(load_path)

    # Generate batch size, learning rate from the hash value
    batch_size, learning_rate, _ = hash_to_architecture(hash_val, referred)  
    # Set the learning rate for the optimizer without resetting the learnable parameters
    referred.optimizer.learning_rate.assign(learning_rate)

    return _training_(x_train, y_train, referred, batch_size, threshold, max_epoch, max_iteration, save_path)

import os
import base64
import tempfile
def model_to_string(model):
    """
    Converts a keras model to a base64-encoded string.

    This function saves a given Keras model to a temporary file, reads the binary
    contents of the file, and encodes the contents to a base64 string. The temporary
    file is deleted after reading.

    Args:
        model: A Keras model instance to be serialized.

    Returns:
        A base64-encoded string representing the keras model.
    """

    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
        temp_file_name = temp_file.name
    
    try:
        # Save the model to the temporary file
        model.save(temp_file_name)
        
        # Read the file's content
        with open(temp_file_name, 'rb') as f:
            model_binary = f.read()
        
        # Encode the binary content to a base64 string
        model_string = base64.b64encode(model_binary).decode('utf-8')
    
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_file_name)
    
    return model_string

# Posthash check for model validity 
def post_check(post_difficulty, nonce, model=None):
    if model is None:
        raise ValueError("Empty input model")

    model_string = model_to_string(model)

    dump = f"{nonce}{model_string}".encode()
    post_hash = hashlib.sha256(dump).hexdigest()
    post_hash_bin = bin(int(post_hash, 16))[2:].zfill(256)
    if post_hash_bin[:post_difficulty] == "0" * post_difficulty:
        b = 1
    else:
        b = 0
    return b, post_hash_bin

# D-PoDL Solver
def dpodl_solver(prev_hash, difficulty, x_train, y_train, threshold, post_difficulty, max_epoch, max_iteration, max_post_check_iteration, save_path, load_path=None, referred=None):
    nonce, hash_val = pre_pow(prev_hash, difficulty)
    if load_path is None and referred is None:
        model = main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path)
    elif load_path is not None and referred is None:
        model = main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path, load_path)
    else:
        model = main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path, None, referred)

    # Perform posthash check
    post_check_iteration = 0
    b, post_hash = post_check(post_difficulty, nonce, model)  # the model is hashed and if below threshold accepted -> introduces certain
    # randomness and that no precomputed solution is possible -> at the same time valid model could be discarded
    while b == 0 and post_check_iteration < max_post_check_iteration:
        # print(f"Posthash check failed with {post_check(post_difficulty, nonce, model)[1]}. Continuing training.")
        model = main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path, None, model)
        b, post_hash = post_check(post_difficulty, nonce, model)
        
    _, accuracy = model.evaluate(x_train, y_train, verbose=0)

    if accuracy >= threshold and b == 1:
        print(f"D-PoDL complete: Model accuracy {accuracy:.4f} with post hash {post_hash}.")
    else:
        print(f"D-PoDL failed: Reached max iterations ({max_iteration, max_post_check_iteration}) with final accuracy {accuracy:.4f}.")

    # Save the trained model using the native Keras format
    model.save(save_path)
    
    return model
