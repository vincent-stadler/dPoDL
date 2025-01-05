from abc import ABC, abstractmethod
import torch
from dPoDL.dpodl_core.predictor_models import FloatSequenceTransformer
import numpy as np


class Predictor(ABC):
    @abstractmethod
    def predict_next_value(self, sequence):
        pass

    @abstractmethod
    def predict_next_values(self, sequence, steps):
        pass


class TransformerPredictor(Predictor):
    def __init__(self, model_path, confidence_threshold):
        self.model = TransformerPredictor.load_model(model_path)
        self.confidence_threshold = confidence_threshold

    def predict_next_value(self, sequence) -> (float, float):
        sequence = sequence[-self.model.input_length:]  # make our sequence have max model.input_length
        sequence = TransformerPredictor.standardize_sequence(sequence)
        self.model.train()  # Set the model to training mode to keep dropout active
        predictions = []

        # Prepare the input sequence
        input_sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(-1)  # Shape: (seq_length, 1)
        padded_input = torch.zeros(self.model.input_length, 1)  # Padding to fixed length
        padded_input[:len(sequence)] = input_sequence  # Copy sequence into padded tensor

        # Create the key_padding_mask
        key_padding_mask = torch.full((1, self.model.input_length), True, dtype=torch.bool)  # All True initially
        key_padding_mask[0, :len(sequence)] = False  # False for valid positions

        # Make the prediction
        padded_input = padded_input.unsqueeze(0)  # Add batch dimension: (1, input_length, 1)

        with torch.no_grad():  # Disable gradient computation during inference
            for _ in range(40):  # 40 times prediction is made with dropout
                prediction = self.model(padded_input, key_padding_mask)
                predictions.append(prediction.item())  # Store the predictions (detached from the computation graph)
        predictions = TransformerPredictor.destandardize_sequence(predictions)
        mean_prediction = predictions.mean(axis=0)  # Mean prediction across samples
        std_prediction = predictions.std(axis=0)  # Standard deviation as uncertainty

        return max(0, mean_prediction), self.derive_confidence_score(std_prediction)  # can't predict negative loss val

    def predict_next_values(self, sequence, steps) -> (list, list):
        std_predictions = [0] * len(sequence)
        previous_input = np.array(sequence)

        for i in range(steps):
            y_hat, std_prediction = self.predict_next_value(previous_input)
            std_predictions.append(std_prediction)
            previous_input = np.concatenate((previous_input, np.array([y_hat])))

        predicted_sequence = previous_input[len(sequence):]
        std_predictions = std_predictions[len(sequence): ]

        return predicted_sequence, std_predictions

    @staticmethod
    def derive_confidence_score(std_prediction):
        normalized_s = std_prediction / FloatSequenceTransformer.MAX_STD_TRAIN  # Normalize by max_std (95th percentile)
        confidence_score = np.exp(-FloatSequenceTransformer.ALPHA * normalized_s)  # Apply exponential decay
        return confidence_score

    @staticmethod
    def load_model(model_path):
        model = FloatSequenceTransformer()
        model.load_state_dict(torch.load(model_path))
        return model

    @staticmethod
    def standardize_sequence(sequence):
        return (np.array(sequence) - FloatSequenceTransformer.TRAINING_MEAN) / FloatSequenceTransformer.TRAINING_STD

    @staticmethod
    def destandardize_sequence(sequence):
        return (np.array(sequence) * FloatSequenceTransformer.TRAINING_STD) + FloatSequenceTransformer.TRAINING_MEAN


if __name__ == '__main__':
    MODEL_PATH = r"C:\Users\daV\Documents\ZHAW\HS 2024\dPoDL\dPoDL\experiments\training\models\cnns_cifar10_categorical\transformer-model_emb8_dropout0.2_layers1_heads1_date05-01-2025.pth"
    FUTURE_STEPS = 5
    predictor = TransformerPredictor(
        model_path=MODEL_PATH,
        confidence_threshold=0.5)
    l = [1.933941125869751, 1.8403935432434082, 1.8168894052505493, 1.8017066717147827, 1.7927669286727905,
         1.7832093238830566, 1.7778414487838745, 1.7706481218338013, 1.7674061059951782,]
    print(predictor.predict_next_values(l, 5))
    adapt confidence score