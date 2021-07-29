import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_algorithm(matrix):
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    matching_mask = np.zeros(matrix.shape, dtype=int)
    matching_mask[rows, cols] = 1
    return matching_mask * matrix, matching_mask