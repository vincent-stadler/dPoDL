import time
import string
import matplotlib.pyplot as plt
import random
from dpodl_core.verification import pre_pow
from dpodl_core.dpodl import dpodl_solver
from models.mnist_task import load_data


def gen_random_string(length=100):
    letters = string.ascii_letters
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


def test_pow(difficulty):
    prev_hash = gen_random_string()
    t1 = time.time()
    pre_pow(prev_hash, difficulty)
    t2 = time.time()
    return t2 - t1


def test_dpodl(difficulty, threshold, post_difficulty, max_epoch, max_iteration, max_post_check_iteration, save_path,
               load_path=None):
    prev_hash = gen_random_string()
    x_train, y_train, _, _ = load_data()

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


def plot(res1, res2):
    plt.figure(figsize=(10, 6))

    plt.plot(res1, label='PoW Runtime', color='blue', linestyle='-', marker='o')
    plt.plot(res2, label='D-PoDL Runtime', color='green', linestyle='--', marker='x')

    plt.xlabel('Number of experiments')
    plt.ylabel('Runtime')
    plt.title('Runtime History')

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    max_epoch = 3
    max_iteration = 4
    max_post_check_iteration = 100
    save_path = "result.keras"
    load_path = None

    pow_results = []
    dpodl_results = []

    for i in range(1):
        print(f"Experiment {i + 1}:")
        pow_results.append(test_pow(4))
        dpodl_results.append(
            test_dpodl(4, 0.98, 2, max_epoch, max_iteration, max_post_check_iteration, save_path, load_path))

    plot(pow_results, dpodl_results)
