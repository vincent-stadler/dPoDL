import random
import numpy as np
from dPoDL.models.model_utils import reset_weights
from dPoDL.dpodl_core.predictor import TransformerPredictor
from dPoDL.dpodl_core.saturation import find_stabilization_point
from dPoDL.models.task_interface import TaskInterface
from dPoDL.callbacks.callbacks import BatchLogger, EarlyStoppingByAccuracy
from typing import Optional, Tuple


MODEL_PATH = r"C:\Users\daV\Documents\ZHAW\HS 2024\dPoDL\dPoDL\experiments\training\models\cnns_cifar10_categorical\transformer-model_emb8_dropout0.2_layers1_heads1_date27-12-2024.pth"
predictor = TransformerPredictor(
    model_path=MODEL_PATH,
    confidence_threshold=0.5)


def hash_to_architecture(hash_val: str, task: TaskInterface):
    bsize = int(hash_val, 16) % len(task.batches)
    lr = int(hash_val, 16) % len(task.lrs)

    # Generate initial learnable parameters by setting a seed
    seed = int(hash_val, 16) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

    task.batch_size, task.learning_rate = task.batches[bsize], task.lrs[lr]


def _training(max_iteration: int, task: TaskInterface):
    for iteration in range(max_iteration):
        task.train(epochs=1)
        current_losses = task.history['loss']
        if find_stabilization_point(current_losses) < len(current_losses):
            print("saturated stopped training")
            break

        next_loss, confidence = predictor.predict_next_value(current_losses)
        if confidence >= predictor.confidence_threshold:
            # check if including the next predicted loss value the saturation point is among the empirically gathered
            # loss values, if yes we return
            if find_stabilization_point(list(current_losses) + [next_loss]) < len(current_losses):
                break

            next_losses, confidences = predictor.predict_next_values(sequence=current_losses, steps=20)
            # hallucinating steps tells us that saturation is reached among current losses, we return
            if find_stabilization_point(list(current_losses) + list(next_losses)) < len(current_losses):
                break
    # out of training loop
    if iteration == max_iteration - 1:
        print("no saturation point found of model training")
    else:
        _, accuracy = task.evaluate()
        print(f"traing saturation point found at iteration {iteration + 1} with accuracy {accuracy:.4f}")
    task.save()


    #batch_logger = BatchLogger()  # Custom batch logger callback
    #early_stopping = EarlyStoppingByAccuracy(threshold=threshold, verbose=0)  # Early stopping callback
    #iteration = 0
    #accuracy = 0
    #while accuracy < threshold and iteration < max_iteration:
    #    print(f"Iteration {iteration + 1}/{max_iteration}")
    #    task.train(max_epoch=max_epoch, callbacks=[batch_logger, early_stopping])
    #    # Evaluate accuracy on the training set
    #    _, accuracy = task.evaluate()
    #    print(f"Current Training Accuracy = {accuracy:.4f}")
    #    iteration += 1
#
    #if accuracy >= threshold:
    #    print(f"Training complete: Model achieved accuracy {accuracy:.4f} which meets or exceeds the threshold of {threshold}.")
    #else:
    #    print(f"Training stopped: Reached max iterations ({max_iteration}) with final accuracy {accuracy:.4f}.")
#
    #task.save()  # Save the trained model


def _training_fresh(hash_val: str, max_iteration: int, task: TaskInterface):

    hash_to_architecture(hash_val, task)
    task.create_model()

    print(f"Hyperparameters of freshly instantiated model are batch_size: {task.batch_size} and learning_rate: {task.learning_rate}")
    return _training( max_iteration, task)


def main_training(hash_val: str, max_iteration: int, task: TaskInterface, load_path: Optional[str] = None):
    if load_path is None and task.model is None:
        return _training_fresh(hash_val, max_iteration, task)

    if load_path and task.model is None:
        # Load the pre-trained model from the given path
        task.load_model(load_path=load_path)
        # Generate batch size, learning rate from the hash value
        hash_to_architecture(hash_val, task)
        # Set the learning rate for the optimizer without resetting the learnable parameters
        task.model.optimizer.learning_rate.assign(task.learning_rate)

    return _training(max_iteration, task)
