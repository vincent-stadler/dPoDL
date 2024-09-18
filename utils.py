from keras.callbacks import Callback

# DEFINITION OF CALLBACKS USED IN TRAINING
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

