import numpy as np
import utils_6
from hungarian_algorithm_6 import hungarian_algorithm

T = 365  # Time horizon
N_EXPS = 1  # Number of experiments

N_ARMS_1 = 5  # Number of different candidate prices
N_ARMS_2 = 5  # Number of candidates for second item price

NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class
SD_CUSTOMERS = np.array([2, 4, 1, 3])  # standard deviation on the number of customers per each class
TOT_CUSTOMERS = np.sum(NUM_CUSTOMERS)

PROMO_PROB = np.array([0.4, 0.2, 0.22, 0.18])
PROMO_DISCOUNTS = np.array([1, 0.85, 0.75, 0.60])

MARGINS_1 = np.linspace(150, 250, N_ARMS_1)
MARGINS_2 = np.multiply(np.linspace(25, 35, N_ARMS_2).reshape((N_ARMS_2, 1)), PROMO_DISCOUNTS.reshape((1, 4)))

CR1 = []
CR2 = []

# constructing matrix of conversion rates for the first product
for margin in MARGINS_1:
    cr = np.array([utils_6.cr1(margin, c_class) for c_class in range(len(NUM_CUSTOMERS))])
    CR1.append(cr)

# constructing matrix of conversion rates for the second product
for margin in MARGINS_2:
    tmp = []
    for c_class in range(len(NUM_CUSTOMERS)):
        cr = np.array([utils_6.cr2(discounted_margin, c_class) for discounted_margin in margin])
        tmp.append(cr)
    CR2.append(tmp)

def compute_opt_matching():
        opt_value = -1
        for arm_1 in range(N_ARMS_1):  # For every price_1
            for arm_2 in range(N_ARMS_2):
                #matching, value = hungarian_algorithm(self.build_matrix(arm_1, arm_2, upper_bound_1, upper_bound_2))
                matching, mask = hungarian_algorithm(build_matrix(arm_1, arm_2))
                value = np.sum(matching)
                if value > opt_value:
                    opt_value = value
                    idx1 = arm_1
                    idx2 = arm_2

        return opt_value  #, (idx1, idx2)

def build_matrix(self, idx1, idx2): 
        n_promos = (PROMO_PROB[1 :] * TOT_CUSTOMERS).astype(int)
        n_promos = np.insert(n_promos, 0, TOT_CUSTOMERS - np.sum(n_promos))

        profit =   CR1 * (MARGINS_1[idx1] + CR2 * MARGINS_2[idx2])

        # repeat columns
        matrix = np.repeat(profit, n_promos, axis=1)

        # repeat rows
        matrix = np.repeat(matrix, NUM_CUSTOMERS, axis=0)

        return matrix


OPT = compute_opt_matching