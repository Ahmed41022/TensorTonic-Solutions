import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)
    if y.size == 0:
        return 0.0
    # Get class frequencies
    _, counts = np.unique(y, return_counts=True)

    # Convert to probabilities
    probs = counts / counts.sum()

    # Remove zero probabilities (numerical stability)
    probs = probs[probs > 0]

    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)