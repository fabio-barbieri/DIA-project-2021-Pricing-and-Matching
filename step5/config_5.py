import numpy as np
from scipy.stats import truncnorm
import sys
from scipy.optimize import linear_sum_assignment

T = 365  # Time horizon

N_EXPS = 20  # Number of experiments

N_ARMS = 20  # Number of different candidate prices

NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class

MARGIN_1 = 200

def compute_cr1(price, cl):
    # MAXIMUM and minimun prices for item 1
    M = 250
    m = 150

    if (price < m) or (price > M): 
        sys.exit('Price not in range')

    # Junior Professional ######################################################################################
    if cl == 0:
        def f(y):
            # Parameters for the first truncated normal
            loc1 = 200
            scale1 = 50
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1

            # Parameters for the second truncated normal
            loc2 = 220
            scale2 = 80
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 

            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(150, 250, 2000)
        ff = f(xx)
        mm = np.argmin(ff)
        MM = np.argmax(ff)
        fmin = f(xx[mm])
        fmax = f(xx[MM])

        return 0.95 * (f(price) - fmin) / (fmax - fmin)

    # Junior Amateur ###########################################################################################
    if cl == 1:
        return np.exp(0.04 * (M - price)) / np.exp(0.04 * (M - m + 2))

    # Senior Professional ######################################################################################
    if cl == 2:
        def g(y):
            # Parameters for the first truncated normal
            loc1 = 200
            scale1 = 60
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1

            # Parameters for the second truncated normal
            loc2 = 230
            scale2 = 40
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 

            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(150,250,2000)
        gg = g(xx)
        mm = np.argmin(gg)
        MM = np.argmax(gg)
        gmin = g(xx[mm])
        gmax = g(xx[MM])

        return 0.95 * (g(price) - gmin) / (gmax - gmin)

    # Senior Amateur ########################################################################################### 
    if cl == 3:
        return np.exp(0.02 * (M - price)) / np.exp(0.02 * (M - m + 2))

CR1 = np.array([compute_cr1(MARGIN_1, c) for c, _ in enumerate(NUM_CUSTOMERS)])

MARGINS_2 = np.array([29.99, 24.99, 20.99, 10.99])
                      # p0     p1     p2     p3

CR2 = np.array([[0.2, 0.4, 0.3, 0.3],  # Junior Professionals
                [0.0, 0.2, 0.3, 0.5],  # Junior Amateur
                [0.1, 0.5, 0.3, 0.1],  # Senior Professionals
                [0.1, 0.1, 0.1, 0.7]]) # Senior Amateur
                # p0   p1   p2   p3

SD_CUSTOMERS = np.array([2, 4, 1, 3])  # standard deviation on the number of customers per each class

PROMO_PROB = np.array([0.4, 0.2, 0.22, 0.18]) # Promo-assignments for each class, fixed by the Business Unit of the shop

def build_matrix(num_customers, promo_prob, cr1, margin_1, cr2, margins_2):
    matrix_dim = np.sum(num_customers)

    # First set integers p1, p2, p3 and the remaining are p0 
    n_promos = (promo_prob[1 :] * matrix_dim).astype(int)
    n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))

    profit = cr1.reshape((4, 1)) * (margin_1 + cr2 * margins_2) 

    # repeat columns
    matrix = np.repeat(profit, n_promos, axis=1)

    # repeat rows
    matrix = np.repeat(matrix, num_customers, axis=0)

    return matrix

def opt_matching(matrix):
    row, col = linear_sum_assignment(matrix, maximize=True)
    matching_mask = np.zeros(matrix.shape, dtype=int)
    matching_mask[row, col] = 1
    return matching_mask * matrix, matching_mask

def compute_opt(num_customers, promo_prob, cr1, margin_1, cr2, margins_2):
    matrix = build_matrix(num_customers, promo_prob, cr1, cr2, margin_1, margins_2)
    return np.sum(opt_matching(matrix))

OPT = compute_opt(NUM_CUSTOMERS, PROMO_PROB, CR1, MARGIN_1, CR2, MARGINS_2)

