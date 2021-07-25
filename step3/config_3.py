import numpy as np
from scipy.stats import truncnorm
import sys

T = 365  # Time horizon

N_EXPS = 20  # Number of experiments

N_ARMS = 20  # Number of different candidate prices

NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class

MARGINS_1 = np.linspace(150, 250, N_ARMS)

def compute_cr1(price, cl):
    # MAXIMUM and minimun prices for item 1
    M = 250
    m = 150
    
    if (price < m) or (price > M): 
        sys.exit('price not in range')

    if cl == 0:       # Junior Professional
        def f(y):
            # parameters for the first truncated normal
            loc1 = 200
            scale1 = 50
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1
            # parameters for the second truncated normal
            loc2 = 220
            scale2 = 80
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 
            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(150,250,2000)
        ff = f(xx)
        mm = np.argmin(ff)
        MM = np.argmax(ff)
        fmin = f(xx[mm])
        fmax = f(xx[MM])
        return 0.95 * (f(price) - fmin) / (fmax - fmin)

    
    if cl == 1:       # Junior Amateur
        return np.exp(0.04 * (M - price)) / np.exp(0.04 * (M - m + 2))

    if cl == 2:       # Senior Professional
        def g(y):
            # parameters for the first truncated normal
            loc1 = 200
            scale1 = 60
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1

            # parameters for the second truncated normal
            loc2 = 230
            scale2 = 40
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 
            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, 2, b2, loc2, scale2)

        xx = np.linspace(150, 250, 2000)
        gg = g(xx)
        mm = np.argmin(gg)
        MM = np.argmax(gg)
        gmin = g(xx[mm])
        gmax = g(xx[MM])
        return 0.95 * (g(price) - gmin) / (gmax - gmin)


    if cl == 3:       # Senior Amateur
        return np.exp(0.02 * (M - price)) / np.exp(0.02 * (M - m + 2))

CR1 = np.array([compute_cr1(m1, c) for m1 in MARGINS_1 for c, _ in enumerate(NUM_CUSTOMERS)]).reshape(len(MARGINS_1), len(NUM_CUSTOMERS))

# CR1 = []
# for margin in MARGINS_1:
#     cr = np.array([utils_3.cr1(margin, c_class) for c_class in range(len(NUM_CUSTOMERS))])
#     CR1.append(cr)

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

def compute_profit(i, cr1, margin1, cr2, margins2, matching, num_customers):
    matching_prob = matching / np.expand_dims(num_customers, axis=1)
    a = cr1[i] * (margin1 + np.dot(cr2 * matching_prob, margins2))  # 4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                    # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                    # = 4x1 * (1x1 + 4x1) = 
                                                                    # = 4x1 * 4x1 = 
                                                                    # = 4x1
    return np.dot(a, num_customers)

known_profits = [compute_profit(i, CR1, m1, CR2, MARGINS_2, MATCHING, NUM_CUSTOMERS,) for i, m1 in enumerate(MARGINS_1)]
OPT = max(known_profits)

