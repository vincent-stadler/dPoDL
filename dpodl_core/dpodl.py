import numpy as np
from dPoDL.dpodl_core.training import main_training
from dPoDL.dpodl_core.verification import post_check, pre_pow
from dPoDL.dpodl_core.saturation import find_stabilization_point
from dPoDL.models.task_interface import TaskInterface
from typing import Optional
from keras.models import Model


def dpodl_solver(prev_hash: str,
                 difficulty: int,
                 post_difficulty: int,
                 max_iteration: int,
                 max_post_check_iteration: int,
                 save_path: str,
                 task: TaskInterface,
                 load_path: Optional[str] = None):
    """
    D-PoDL solver that manages the Proof-of-Work (PoW) and post-hash validation.

    Args:
        prev_hash (str): Previous block's hash for PoW.
        difficulty (int): Difficulty level for PoW.
        post_difficulty (int): Difficulty for post-hash validation.
        max_iteration (int): Maximum iterations for training until desired accuracy is reached.
        max_post_check_iteration (int): Maximum iterations for post-hash validation.
        save_path (str): Path to save the trained model.
        task (TaskInterface): Task interface
        load_path (Optional[str]): Path to a pre-trained model to load.

    Returns:
        Model: The trained Keras model after solving.
    """
    # The pre PoW is performed according to the "difficulty" parameter
    nonce, hash_val = pre_pow(prev_hash, difficulty)
    task.load_data()
    # Model is initialized and runs through first training cycle.
    # If there already exists a model, it will be used for the training.
    if load_path is None and task.model is None:
        # New model is instantiated and being trained
        main_training(hash_val, max_iteration, task)
    elif load_path is not None and task.model is None:
        # Existing model is loaded from disk and being trained
        main_training(hash_val, max_iteration, task, load_path)
    elif isinstance(task.model, Model):
        # Existing model that is saved in "referred" variable is being trained
        main_training(hash_val, max_iteration, task)
    else:
        raise ValueError("Check 'referred' and 'load_path' parameters are adequate")

    # Perform posthash verification
    post_check_iteration = 0
    valid, post_hash = post_check(post_difficulty, nonce, task.model)
    # The model is hashed and accepted if it's below "threshold" parameter. If the hash is below the desired threshold
    # another training cycle is initiated. This process is repeated "max_post_check_iteration" times. Thus, certain
    # randomness is introduced and  ensures that no precomputed solution is possible.
    # At the same time a valid model could be discarded, and has to train at least one more epoch (or batch if the
    # callback EarlyStoppingByAccuracy is replaced by the  EarlyStoppingByBatchAccuracy callback)
    easier_max_iteration = max_iteration
    while valid == 0 and post_check_iteration < max_post_check_iteration:
        print("Post hash verification failed. Continuing to train model")
        easier_max_iteration = max(1, easier_max_iteration //2) # half max iterations time, at least 1
        main_training(hash_val, max_iteration, task)
        print("Performing post hash verification of trained model")
        valid, post_hash = post_check(post_difficulty, nonce, task.model)
        post_check_iteration += 1

    _, accuracy = task.evaluate()
    if find_stabilization_point(task.history["loss"]) < len(task.history["loss"]) or find_stabilization_point(task.history["val_loss"]) < len(task.history["val_loss"]):
        print(f"D-PoDL complete: Model accuracy {accuracy:.4f} with valid post hash {post_hash}.")
    else:
        print(f"D-PoDL failed: Reached max iterations (max_iteration: {max_iteration}, max_post_check_iteration: {max_post_check_iteration}) with final accuracy {accuracy:.4f}.")
    return

    #if accuracy >= threshold and valid == 1:
    #    print(f"D-PoDL complete: Model accuracy {accuracy:.4f} with valid post hash {post_hash}.")
    #else:
    #    print(f"D-PoDL failed: Reached max iterations (max_iteration: {max_iteration}, max_post_check_iteration: {max_post_check_iteration}) with final accuracy {accuracy:.4f}. Desired accuracy threshold is at {threshold}")

    #task.plot_metrics()
