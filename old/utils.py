from keras.callbacks import Callback
from keras.models import Sequential, clone_model, load_model
import keras

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



# RESET MODEL WEIGHTS
def reset_weights(model):
    # Clone the model to create a new instance with the same architecture
    new_model = clone_model(model)
    new_model.set_weights(model.get_weights())  # Initialize with the same weights

    for layer in new_model.layers:
        # Check if the layer has weights (some layers like Dropout do not have weights)
        if isinstance(layer, keras.layers.Layer) and layer.get_weights():
            # Initialize the kernel weights
            if hasattr(layer, "kernel_initializer"):
                kernel_shape = layer.kernel.shape
                kernel = layer.kernel_initializer(kernel_shape)
            else:
                kernel = None

            # Initialize the bias weights
            if hasattr(layer, "bias_initializer") and layer.bias is not None:
                bias_shape = layer.bias.shape
                bias = layer.bias_initializer(bias_shape)
            else:
                bias = None

            # Set the weights of the layer
            if kernel is not None and bias is not None:
                layer.set_weights([kernel, bias])
            elif kernel is not None:
                layer.set_weights([kernel])

    return new_model