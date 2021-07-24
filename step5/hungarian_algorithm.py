import numpy as np
import copy
from scipy.optimize import linear_sum_assignment

def hungarian_algorithm(matrix):
    m = np.max(matrix) - matrix
    row, col = linear_sum_assignment(m)
    matching_mask = np.zeros(matrix.shape, dtype=int)
    matching_mask[row, col] = 1
    return matching_mask * matrix, matching_mask
