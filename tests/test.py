import time
import string
import matplotlib.pyplot as plt
import random
from dpodl_core.verification import pre_pow
from dpodl_core.dpodl import dpodl_solver
from models.mnist_task import MNISTtask
from typing import Optional, List


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


def test_dpodl(difficulty: int, threshold: float, post_difficulty: int, max_epoch: int, max_iteration: int,
               max_post_check_iteration: int, save_path: str, load_path: Optional[str] = None) -> float:
    """
    Tests the D-PoDL (Distributed Proof of Deep Learning) solver with the given parameters.

    Args:
        difficulty (int): The difficulty level for the D-PoDL algorithm.
        threshold (float): The accuracy threshold for stopping.
        post_difficulty (int): Post-check difficulty level.
        max_epoch (int): Maximum number of epochs per training iteration.
        max_iteration (int): Maximum number of training iterations.
        max_post_check_iteration (int): Maximum number of post-check iterations.
        save_path (str): Path to save the final model.
        load_path (Optional[str]): Path to load a pre-trained model. Defaults to None.

    Returns:
        float: The time taken to complete the D-PoDL process in seconds.
    """
    task = MNISTtask()  # for this test we use the MNIST task
    prev_hash = gen_random_string()
    x_train, y_train, _, _ = task.load_data()

    t1 = time.time()
    dpodl_solver(
        prev_hash=prev_hash,
        difficulty=difficulty,
        x_train=x_train,
        y_train=y_train,
        threshold=threshold,
        post_difficulty=post_difficulty,
        max_epoch=max_epoch,
        max_iteration=max_iteration,
        max_post_check_iteration=max_post_check_iteration,
        save_path=save_path,
        load_path=load_path
    )
    t2 = time.time()

    return t2 - t1


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


if __name__ == "__main__":
    max_epoch = 3
    max_iteration = 4
    max_post_check_iteration = 10
    save_path = "result.keras"
    load_path = None

    pow_results: List[float] = []
    dpodl_results: List[float] = []

    for i in range(2):
        print(f"Experiment {i + 1}:")
        pow_results.append(test_pow(4))
        dpodl_results.append(
            test_dpodl(10, 0.98, 2, max_epoch, max_iteration, max_post_check_iteration, save_path, load_path)
        )

    plot(pow_results, dpodl_results)
