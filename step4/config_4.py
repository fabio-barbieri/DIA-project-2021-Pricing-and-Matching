import numpy as np
import utils_4

T = 365  # Time horizon
N_EXPS = 20  # Number of experiments
N_ARMS = 20  # Number of different candidate prices
NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class

MARGINS_1 = np.linspace(150, 250, N_ARMS)

CR1 = []
for margin in MARGINS_1:
    cr = np.array([utils_4.cr1(margin, c_class) for c_class in range(len(NUM_CUSTOMERS))])
    CR1.append(cr)

MATCHING = np.array([[8,  5, 4,  3],  # Class 1 -> tot = NUM_CUSTOMERS[0]
                     [16, 6, 10, 8],  # Class 2 -> tot = NUM_CUSTOMERS[1]
                     [2,  3, 3,  2],  # Class 3 -> tot = NUM_CUSTOMERS[2]
                     [14, 6, 5,  5]]) # Class 4 -> tot = NUM_CUSTOMERS[3]
                    # p0  p1 p2  p3

MARGINS_2 = np.array([29.99, 24.99, 20.99, 10.99])
                    # p0     p1     p2     p3

CR2 = np.array([[0.2, 0.4, 0.3, 0.3],  # Junior Professionals
                [0.0, 0.2, 0.3, 0.5],  # Junior Amateur
                [0.1, 0.5, 0.3, 0.1],  # Senior Professionals
                [0.1, 0.1, 0.1, 0.7]]) # Senior Amateur
                # p0   p1   p2   p3

weighted_averages = []
for i, arm in enumerate(MARGINS_1):  # For every price_1
    arm_expected_profit = 0
    for j, n_customers in enumerate(NUM_CUSTOMERS):  # For every customer class
        exp_buyers_item1 = n_customers * CR1[i][j]
        margin1 = arm
        promo_assigment_prob = MATCHING[j, :] / n_customers
        margin2 = np.multiply(MARGINS_2, CR2[j, :])

        arm_expected_profit += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))
    weighted_averages.append(arm_expected_profit)

OPT = np.max(weighted_averages)