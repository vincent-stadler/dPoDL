import hashlib
from models.model_utils import model_to_string
from typing import Tuple, Optional
from keras import Model


def pre_pow(prev_hash: str, difficulty: int) -> Tuple[int, str]:
    """
    Perform Proof-of-Work (PoW) by finding a valid nonce.

    Args:
        prev_hash (str): Previous block's hash.
        difficulty (int): Difficulty level for PoW.

    Returns:
        Tuple[int, str]: Nonce and corresponding hash value.
    """
    nonce = 0
    while True:
        combined = f"{prev_hash}{nonce}".encode()
        new_hash = hashlib.sha256(combined).hexdigest()
        new_hash_bin = bin(int(new_hash, 16))[2:].zfill(256) #zfill ensures the binary string always represents 256 bits
        if new_hash_bin[:difficulty] == "0" * difficulty:  # checking if satisfies given threshold
            return nonce, new_hash_bin
        nonce += 1


def post_check(post_difficulty: int, nonce: int, model: Optional[Model] = None) -> Tuple[int, str]:
    """
    Perform a post-hash check to validate the trained model.

    Args:
        post_difficulty (int): Difficulty level for post-hash validation.
        nonce (int): Nonce generated in PoW.
        model (Optional[Model]): Keras model to be validated.

    Returns:
        Tuple[int, str]: Post-hash validity bit (1 for valid, 0 for invalid) and the corresponding post-hash value.
    """
    if model is None:
        raise ValueError("Empty input model")

    model_string = model_to_string(model)

    dump = f"{nonce}{model_string}".encode()
    post_hash = hashlib.sha256(dump).hexdigest()
    post_hash_bin = bin(int(post_hash, 16))[2:].zfill(256)
    if post_hash_bin[:post_difficulty] == "0" * post_difficulty:
        b = 1
    else:
        b = 0
    return b, post_hash_bin
