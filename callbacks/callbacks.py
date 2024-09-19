from keras.callbacks import Callback
from typing import Optional


class BatchLogger(Callback):
    """
    Custom Keras callback to log information at the end of each batch.
    """
    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """
        Override the Keras method to log information after each batch ends.

        Args:
            batch (int): Batch index.
            logs (Optional[dict]): Logs for the batch (e.g., loss, accuracy).
        """
        pass


class EarlyStoppingByBatchAccuracy(Callback):
    """
    Custom Keras callback to stop training early if accuracy exceeds the threshold within a batch.
    """
    def __init__(self, threshold: float, monitor: str = "accuracy", verbose: int = 0):
        super(EarlyStoppingByBatchAccuracy, self).__init__()
        self.threshold = threshold
        self.monitor = monitor
        self.verbose = verbose

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """
        Override the method to stop training if accuracy reaches the threshold.

        Args:
            batch (int): Batch index.
            logs (Optional[dict]): Logs for the batch (e.g., loss, accuracy).
        """
        accuracy = logs.get(self.monitor) if logs else None
        if accuracy is not None and accuracy >= self.threshold:
            self.model.stop_training = True
            if self.verbose > 0:
                print(f"\nBatch {batch + 1}: early stopping because {self.monitor} reached {accuracy:.4f}")


class EarlyStoppingByAccuracy(Callback):
    """
    Custom Keras callback to stop training early if accuracy exceeds the threshold at the end of an epoch.
    """
    def __init__(self, threshold: float, monitor: str = "accuracy", verbose: int = 0):
        super(EarlyStoppingByAccuracy, self).__init__()
        self.threshold = threshold
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """
        Override the method to stop training if accuracy reaches the threshold at the end of an epoch.

        Args:
            epoch (int): Epoch index.
            logs (Optional[dict]): Logs for the epoch (e.g., loss, accuracy).
        """
        accuracy = logs.get(self.monitor) if logs else None
        if accuracy is not None and accuracy >= self.threshold:
            self.model.stop_training = True
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: early stopping because {self.monitor} reached {accuracy:.4f}")
