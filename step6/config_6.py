import numpy as np
import utils_4

T = 365  # Time horizon
N_EXPS = 1  # Number of experiments
N_ARMS = 1  # Number of different candidate prices
NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class
SD_CUSTOMERS = np.array([2, 4, 1, 3])  # standard deviation on the number of customers per each class
TOT_CUSTOMERS = np.sum(NUM_CUSTOMERS)

MARGINS_1 = np.linspace(150, 250, N_ARMS)

CR1 = []
for margin in MARGINS_1:
    cr = np.array([utils_4.cr1(margin, c_class) for c_class in range(len(NUM_CUSTOMERS))])
    CR1.append(cr)

PROMO_PROB = np.array([0.4, 0.2, 0.22, 0.18])

# fixed assigments of promos by the business unit: [0.40, 0.20, 0.22, 0.18]
# MATCHING_PROB[i,j] of the TOT_CUSTOMERS is of class i and receives Pj
MATCHING_PROB = np.array([[0.08, 0.05, 0.04, 0.03],  # Class 1
                    	  [0.16, 0.06, 0.10, 0.08],  # Class 2 
                     	  [0.02, 0.03, 0.03, 0.02],  # Class 3 
                     	  [0.14, 0.06, 0.05, 0.05]]) # Class 4 
#                    	   p0     p1    p2    p3


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
        promo_assigment_prob = MATCHING_PROB[j, :] / n_customers * TOT_CUSTOMERS 
        margin2 = np.multiply(MARGINS_2, CR2[j, :])

        arm_expected_profit += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))
    weighted_averages.append(arm_expected_profit)

OPT = np.max(weighted_averages)
