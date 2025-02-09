# Proof-of-Useful-Work Framework

## Introduction
Traditional Proof-of-Work systems consume excessive energy without producing useful computations. This project explores Proof-of-Useful-Work (PoUW), specifically Proof-of-Deep-Learning (PoDL), where mining power is used for training deep learning models instead of solving cryptographic puzzles.

A key challenge in PoDL is identifying when training remains beneficial. This framework integrates a Transformer-based predictor and a stabilization algorithm to detect loss saturation points, optimizing resource allocation by stopping inefficient training. Experimental results show improved computational efficiency and adaptability compared to static threshold approaches.
## About This Repository
This repository provides a test pipeline for the Proof-of-Useful-Work framework. The test script `test.py` allows users to evaluate the pipeline by running PoW and distributed proof-of-deep-learning tasks. The modular architecture enables flexibility in swapping predictor models and tasks, making it highly customizable.

## Parameters in `test.py`
The following parameters in `test.py` control the execution of the pipeline:

- `difficulty = 0` – Defines the pre-hash difficulty level for the PoW verification process. Set to 0 for testing purposes but can be adjusted to also influence block time.
- `post_difficulty = 0` – Specifies the post-check difficulty level for D-PoDL verification. Set to 0 for testing purposes but can be adjusted to also influence block time.
- `max_iteration = 100` – The maximum number of training iterations for the deep learning task.
- `max_post_check_iteration = 10` – The maximum number of post-check iterations for verifying the model's integrity.

## Modularity and Extensibility
### Process Flow
Below is an image illustrating the core procedure, where the predictor forecasts loss values and the stabilization detection algorithm determines whether the saturation point of the (validation) loss function has been reached.

![Process Flow](Flowchart-dPoDL.png)

### Predictors
In the `dpodl_core` folder, the `predictor.py` file defines the predictor used for forecasting loss values. This predictor can be modified or swapped with different models to test alternative forecasting approaches.

### Tasks
Tasks such as `MNISTtask` and `CIFAR10task` are implemented as modular components. Users can swap or extend tasks as long as they implement the base abstract class `TaskInterface`. This ensures seamless integration of new deep learning tasks into the D-PoDL pipeline.

## Running the Test Pipeline
To test the pipeline, first install the required dependencies by running:
```bash
pip install -r requirements.txt
```
Then, execute the test script with:
```bash
python test.py
```
This will execute the PoW and D-PoDL tasks, logging runtimes and generating performance plots. At the end, two plots will be produced, showing the trained models' accuracy and loss over time. Additionally, during the entire training process, console outputs will provide insights into the progress.
