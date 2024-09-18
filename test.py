import argparse
import time
import string
import random
from dpodl import pre_pow, dpodl_solver, load_data

def gen_random_string(length=100):
    letters = string.ascii_letters
    result = ''.join(random.choice(letters) for i in range(length))

    return result

def test_pow(difficulty):
    prev_hash = gen_random_string()
    t1 = time.time()
    pre_pow(prev_hash, difficulty)
    t2 = time.time()
    # print (f"pow result: {hash}")

    return t2 - t1

def test_dpodl(difficulty, threshold, post_difficulty):
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Test D-PoDL Solver")
    parser.add_argument('--max-epoch', type=int, default=3, help="Maximum epochs per training iteration")
    parser.add_argument('--max-iteration', type=int, default=4, help="Maximum iterations for training")
    parser.add_argument('--max-post-check-iteration', type=int, default=100, help="Maximum post-check iterations")
    parser.add_argument('--save-path', type=str, default="result.keras", help="Path to save the final model")
    parser.add_argument('--load-path', type=str, help="Path to load a pre-trained model")
    args = parser.parse_args()

    prev_hash = gen_random_string()
    x_train, y_train, _, _ = load_data()

    # Run the D-PoDL Solver
    t1 = time.time()
    dpodl_solver(
        prev_hash=prev_hash,
        difficulty=difficulty,
        x_train=x_train,
        y_train=y_train,
        threshold=threshold,
        post_difficulty=post_difficulty,
        max_epoch=args.max_epoch,
        max_iteration=args.max_iteration,
        max_post_check_iteration=args.max_post_check_iteration,
        save_path=args.save_path,
        load_path=args.load_path
    )
    t2 = time.time()

    return t2 - t1

pow_results = []
dpodl_results = []
for i in range(10):
    print (f"Experiment {i+1}:")
    pow_results.append(test_pow(28))
    dpodl_results.append(test_dpodl(4, 0.98, 2))

import matplotlib.pyplot as plt
def plot(res1, res2):
    plt.figure(figsize=(10, 6))
    
    # Plotting the total objective history
    plt.plot(res1, label='PoW Runtime', color='blue', linestyle='-', marker='o')
    plt.plot(res2, label='D-PoDL Runtime', color='green', linestyle='--', marker='x')

    # Adding labels and title
    plt.xlabel('Number of experiments')
    plt.ylabel('Runtime')
    plt.title('Runtime History')
    
    # Adding legend
    plt.legend()
    # Adding grid
    plt.grid(True)
    # Show the plot
    plt.show()

plot(pow_results, dpodl_results)
