import keras
from keras.models import clone_model
import os
import base64
import tempfile
from keras.models import Model


def model_to_string(model: Model) -> str:
    """
    Converts a keras model to a base64-encoded string.

    This function saves a given Keras model to a temporary file, reads the binary
    contents of the file, and encodes the contents to a base64 string. The temporary
    file is deleted after reading.

    Args:
        model: A Keras model instance to be serialized.

    Returns:
        A base64-encoded string representing the keras model.
    """

    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
        temp_file_name = temp_file.name

    try:
        # Save the model to the temporary file
        model.save(temp_file_name)
        # Read the file's content
        with open(temp_file_name, 'rb') as f:
            model_binary = f.read()
        # Encode the binary content to a base64 string
        model_string = base64.b64encode(model_binary).decode('utf-8')
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_file_name)

    return model_string


def reset_weights(model: Model) -> Model:
    """
    Clones a given Keras model and reinitializes its weights.

    This function creates a new instance of the same model architecture and reinitializes
    its weights. It checks if each layer has weights, and if applicable, resets the kernel
    and bias weights using their respective initializers.

    Args:
        model (Model): The Keras model to clone and reset.

    Returns:
        Model: A new Keras model instance with the same architecture and reset weights.
    """
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
