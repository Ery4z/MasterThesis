import numpy as np
import pickle

def slog(data):
    """Utility function to get the log value without loosing the sign

    Args:
        data (np.array): array

    Returns:
        np.array: array
    """
    return np.nan_to_num(np.log(np.abs(data))*np.sign(data))

def load_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    