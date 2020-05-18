"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np
from sklearn.metrics import jaccard_similarity_score as jsc

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    inters = sum([1 if (a[x, y, z] != 0 and b[x, y, z] != 0) else 0 for x in range(a.shape[0]) for y in range(a.shape[1]) for z in range(a.shape[2])])
    total_volumes = sum(np.ones(a[a != 0].shape)) + sum(np.ones(b[b != 0].shape))

    if total_volumes == 0:
        return -1

    return 2.*float(inters) / float(total_volumes)

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # <YOUR CODE GOES HERE>
    x = np.asarray(a, np.bool)
    y = np.asarray(b, np.bool)
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

