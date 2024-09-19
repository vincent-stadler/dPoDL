import numpy as np
from dpodl_core.training import main_training
from dpodl_core.verification import post_check, pre_pow
from typing import Optional
from keras.models import Model


def dpodl_solver(prev_hash: str,
                 difficulty: int,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 threshold: float,
                 post_difficulty: int,
                 max_epoch: int,
                 max_iteration: int,
                 max_post_check_iteration: int,
                 save_path: str,
                 load_path: Optional[str] = None,
                 referred: Optional[Model] = None) -> Model:
    """
    D-PoDL solver that manages the Proof-of-Work (PoW) and post-hash validation.

    Args:
        prev_hash (str): Previous block's hash for PoW.
        difficulty (int): Difficulty level for PoW.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Labels for training data.
        threshold (float): Desired training accuracy.
        post_difficulty (int): Difficulty for post-hash validation.
        max_epoch (int): Maximum epochs per iteration.
        max_iteration (int): Maximum iterations for training.
        max_post_check_iteration (int): Maximum iterations for post-hash validation.
        save_path (str): Path to save the trained model.
        load_path (Optional[str]): Path to a pre-trained model to load.
        referred (Optional[Model]): Referred model if available.

    Returns:
        Model: The trained Keras model after solving.
    """
    nonce, hash_val = pre_pow(prev_hash, difficulty)
    if load_path is None and referred is None:
        model = main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path)
    elif load_path is not None and referred is None:
        model = main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path, load_path)
    else:
        model = main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path, None, referred)

    # Perform posthash check
    post_check_iteration = 0
    b, post_hash = post_check(post_difficulty, nonce, model)
    while b == 0 and post_check_iteration < max_post_check_iteration:
        model = main_training(hash_val, x_train, y_train, threshold, max_epoch, max_iteration, save_path, None, model)
        b, post_hash = post_check(post_difficulty, nonce, model)

    _, accuracy = model.evaluate(x_train, y_train, verbose=0)

    if accuracy >= threshold and b == 1:
        print(f"D-PoDL complete: Model accuracy {accuracy:.4f} with post hash {post_hash}.")
    else:
        print(f"D-PoDL failed: Reached max iterations ({max_iteration, max_post_check_iteration}) with final accuracy {accuracy:.4f}.")

    model.save(save_path)
    return model
