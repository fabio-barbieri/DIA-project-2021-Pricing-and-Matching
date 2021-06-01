import numpy as np
import utils

# Fix the seed for numpy in order to redo experiments
np.random.seed(1234)

T = 365  # Time horizon
N_EXPS = 50  # Number of experiments
N_ARMS = 5  # Number of different candidate prices
NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class

MARGINS_1 = np.linspace(150, 250, 20)

CR1 = []
for margin in MARGINS_1:
    cr = np.array([utils.cr1(margin, c_class) for c_class in range(len(NUM_CUSTOMERS))])
    CR1.append(cr)

weighted_averages = []

for cr_row in CR1:
    weighted_averages.append(np.dot(cr_row, NUM_CUSTOMERS) / sum(NUM_CUSTOMERS))

OPT = CR1[np.argmax(weighted_averages)]

MATCHING = np.array([[8,  5, 4,  3],  # Class 1 -> tot = NUM_CUSTOMERS[0]
                     [16, 6, 10, 8],  # Class 2 -> tot = NUM_CUSTOMERS[1]
                     [2,  3, 3,  2],  # Class 3 -> tot = NUM_CUSTOMERS[2]
                     [14, 6, 5,  5]]) # Class 4 -> tot = NUM_CUSTOMERS[3]
                    # p0   p1   p2   p3

MARGINS_2 = np.array([29.99, 24.99, 20.99, 10.99])
                    # p0     p1     p2     p3

CONV_RATES_2 = np.array([[0.2, 0.4, 0.3, 0.3],  # Junior Professionals
                         [0.0, 0.2, 0.3, 0.5],  # Junior Amateur
                         [0.1, 0.5, 0.3, 0.1],  # Senior Professionals
                         [0.1, 0.1, 0.1, 0.7]]) # Senior Amateur
                        # p0   p1   p2   p3