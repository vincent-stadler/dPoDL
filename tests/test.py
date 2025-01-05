import time
import string
import random
from dPoDL.dpodl_core.verification import pre_pow
from dPoDL.dpodl_core.dpodl import dpodl_solver
from dPoDL.models.mnist_task import MNISTtask
import matplotlib.pyplot as plt
import os
from typing import Optional, List
from tensorflow import get_logger


def gen_random_string(length: int = 100) -> str:
    """
    Generates a random string of the specified length using letters from the ASCII alphabet.

    Args:
        length (int): Length of the random string. Defaults to 100.

    Returns:
        str: A random string of the given length.
    """
    letters = string.ascii_letters
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


def test_pow(difficulty: int) -> float:
    """
    Tests the proof of work (PoW) process with the given difficulty.

    Args:
        difficulty (int): The difficulty level for the PoW algorithm.

    Returns:
        float: The time taken to complete the PoW process in seconds.
    """
    prev_hash = gen_random_string()
    t1 = time.time()
    pre_pow(prev_hash, difficulty)
    t2 = time.time()
    return t2 - t1


def test_dpodl(difficulty: int, post_difficulty: int, max_iteration: int,
               max_post_check_iteration: int, save_path: str, load_path: Optional[str] = None):
    """
    Tests the D-PoDL (Distributed Proof of Deep Learning) solver with the given parameters.

    Args:
        difficulty (int): The difficulty level for the D-PoDL algorithm.
        post_difficulty (int): Post-check difficulty level.
        max_iteration (int): Maximum number of training iterations.
        max_post_check_iteration (int): Maximum number of post-check iterations.
        save_path (str): Path to save the final model.
        load_path (Optional[str]): Path to load a pre-trained model. Defaults to None.

    Returns:
        float: The time taken to complete the D-PoDL process in seconds.
    """
    task = MNISTtask(save_path)  # for this test we use the MNIST task
    prev_hash = gen_random_string()

    t1 = time.time()
    dpodl_solver(
        prev_hash=prev_hash,
        difficulty=difficulty,
        post_difficulty=post_difficulty,
        max_iteration=max_iteration,
        max_post_check_iteration=max_post_check_iteration,
        save_path=save_path,
        load_path=load_path,
        task=task
    )
    t2 = time.time()

    return t2 - t1, task


def plot(res1: List[float], res2: List[float]) -> None:
    """
    Plots the runtime history of PoW and D-PoDL algorithms.

    Args:
        res1 (List[float]): List of runtimes for the PoW process.
        res2 (List[float]): List of runtimes for the D-PoDL process.
    """
    plt.figure(figsize=(10, 6))

    x_values = range(1, len(res1) + 1)
    plt.plot(x_values, res1, label='PoW Runtime', color='blue', linestyle='-', marker='o')
    plt.plot(x_values, res2, label='D-PoDL Runtime', color='green', linestyle='--', marker='x')

    plt.xlabel('Experiment number')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime History')

    plt.legend()
    plt.grid(True)
    plt.xticks(x_values)
    plt.show()

def plot_aggregate_histories(histories, save_path='aggregate_metrics.png'):
    plt.figure(figsize=(12, 5))

    # Get the colormap
    colormap = plt.colormaps['plasma']

    epoch_range = range(1, len(histories[0].get('accuracy', [])) + 1)

    # Plotting Training Accuracy
    plt.subplot(1, 2, 1)
    for idx, history in enumerate(histories):
        color = colormap(idx / len(histories))  # Generate a color for each model
        if 'accuracy' in history:
            plt.plot(epoch_range, history['accuracy'], label=f'Model {idx + 1} Training', color=color, linestyle='-')
        if 'val_accuracy' in history:
            plt.plot(epoch_range, history['val_accuracy'], label=f'Model {idx + 1} Validation', color=color, linestyle='--')
    plt.title('Aggregate Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting Training Loss
    plt.subplot(1, 2, 2)
    for idx, history in enumerate(histories):
        color = colormap(idx / len(histories))  # Generate a color for each model
        if 'loss' in history:
            plt.plot(epoch_range, history['loss'], label=f'Model {idx + 1} Training', color=color, linestyle='-')
        if 'val_loss' in history:
            plt.plot(epoch_range, history['val_loss'], label=f'Model {idx + 1} Validation', color=color, linestyle='--')
    plt.title('Aggregate Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #no warning
    get_logger().setLevel('ERROR')
    difficulty = 0
    post_difficulty = 0
    max_iteration = 100
    max_post_check_iteration = 10
    save_path = "result.keras"
    load_path = None

    pow_results: List[float] = []
    dpodl_results: List[float] = []
    dpodl_models: List[float] = []

    for i in range(1):
        print(f"Experiment {i + 1}:")
        pow_results.append(test_pow(4))
        t, task = test_dpodl(difficulty=difficulty, post_difficulty=post_difficulty, max_iteration=max_iteration,
                             max_post_check_iteration=max_post_check_iteration, save_path=save_path)
        dpodl_results.append(t)
        dpodl_models.append(task.history)


    #plot(pow_results, dpodl_results)
    #plot_aggregate_histories(dpodl_models)