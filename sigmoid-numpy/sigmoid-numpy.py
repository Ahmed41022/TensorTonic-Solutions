import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    values = np.asarray(x, dtype=float)
    results = 1 / (1 + np.exp(-values))
        
    return results