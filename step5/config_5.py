import numpy as np
import utils_5

T = 10  # Time horizon
N_EXPS = 1  # Number of experiments
NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class
SD_CUSTOMERS = np.array([2, 4, 1, 3])  # standard deviation on the number of customers per each class
TOT_CUSTOMERS = np.sum(NUM_CUSTOMERS)

MARGIN_1 = 200

CR1 = np.array([utils_5.cr1(MARGIN_1, c_class) for c_class in range(len(NUM_CUSTOMERS))])

# fixed fractions of promotions assigned by the marketing unit
PROMO_PROB = np.array([0.4, 0.2, 0.22, 0.18])

MARGINS_2 = np.array([29.99, 24.99, 20.99, 10.99])
                      # p0     p1     p2     p3

CR2 = np.array([[0.2, 0.4, 0.3, 0.3],  # Junior Professionals
                [0.0, 0.2, 0.3, 0.5],  # Junior Amateur
                [0.1, 0.5, 0.3, 0.1],  # Senior Professionals
                [0.1, 0.1, 0.1, 0.7]]) # Senior Amateur
                # p0   p1   p2   p3

OPT = utils_5.build_optimal_matching(NUM_CUSTOMERS, PROMO_PROB, CR1, CR2, MARGIN_1, MARGINS_2)

