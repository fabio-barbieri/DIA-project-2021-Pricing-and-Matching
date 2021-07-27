import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_algorithm(matrix):
    row, col = linear_sum_assignment(matrix, maximize=True)
    matching_mask = np.zeros(matrix.shape, dtype=int)
    matching_mask[row, col] = 1
    return row, col, matching_mask * matrix, matching_mask
