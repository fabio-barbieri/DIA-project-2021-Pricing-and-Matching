import numpy as np
import utils_6

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

