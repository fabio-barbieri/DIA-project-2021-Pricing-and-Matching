import numpy as np
from scipy.stats import truncnorm
import sys
from scipy.optimize import linear_sum_assignment

SETTING = int(input('Specify the setting for the current experiment (0 or 1): '))
while (SETTING != 0) and (SETTING != 1):
    print('Wrong setting, try again!')
    SETTING = int(input('Specify the setting for the current experiment (0 or 1): '))

T = 365  # Time horizon

N_EXPS = 1  # Number of experiments

N_ARMS_1 = 5  # Number of different candidate prices for item 1

N_ARMS_2 = 5  # Number of different candidate prices for item 2

NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class

MARGINS_1 = np.linspace(100, 150, N_ARMS_1)

def compute_cr1(price, cl):
    # MAXIMUM and minimun prices for item 1
    M = 150
    m = 100

    if (price < m) or (price > M): 
        sys.exit('Price not in range')

    # Junior Professional ######################################################################################
    if cl == 0:
        def f(y):
            # Parameters for the first truncated normal
            loc1 = 125
            scale1 = 25
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1

            # Parameters for the second truncated normal
            loc2 = 140
            scale2 = 40
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 

            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(100, 150, 1000)
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
            loc1 = 125
            scale1 = 30
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1

            # Parameters for the second truncated normal
            loc2 = 140
            scale2 = 20
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 

            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(100, 150, 1000)
        gg = g(xx)
        mm = np.argmin(gg)
        MM = np.argmax(gg)
        gmin = g(xx[mm])
        gmax = g(xx[MM])

        return 0.95 * (g(price) - gmin) / (gmax - gmin)

    # Senior Amateur ########################################################################################### 
    if cl == 3:
        return np.exp(0.02 * (M - price)) / np.exp(0.02 * (M - m + 2))

CR1 = np.array([compute_cr1(m1, c) for m1 in MARGINS_1 for c, _ in enumerate(NUM_CUSTOMERS)]).reshape(len(MARGINS_1), len(NUM_CUSTOMERS))

#                           p0  p1    p2    p3  
promo_discounts = np.array([1, 0.85, 0.75, 0.60])
MARGINS_2 = np.linspace(25, 35, N_ARMS_2).reshape((N_ARMS_2, 1)) * promo_discounts.reshape((1, 4))

def compute_cr2(discounted_price, cl):
	# MAXIMUM and minimun prices for item 2
	M = 37
	m = 12

	if discounted_price < m or discounted_price > M: 
		sys.exit('discounted_price not in range')

    # Junior Professional ######################################################################################
	if cl == 0:
		def f(y):
			# parameters for the first truncated normal
			loc1 = 25
			scale1 = 5
			a1 = (m - loc1) / scale1
			b1 = (M - loc1) / scale1

			# parameters for the second truncated normal
			loc2 = 29
			scale2 = 8
			a2 = (m - loc2) / scale2
			b2 = (M - loc2) / scale2 

			return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

		xx = np.linspace(12, 37, 1000)
		ff = f(xx)
		mm = np.argmin(ff)
		MM = np.argmax(ff)
		fmin = f(xx[mm])
		fmax = f(xx[MM])

		return 0.95 * (f(discounted_price) - fmin) / (fmax - fmin)

    # Junior Amateur ###########################################################################################
	if cl == 1:
		return np.exp(0.04 * (M - discounted_price)) / np.exp(0.04 * (M - m + 2))
		
    # Senior Professional ######################################################################################
	if cl == 2:
		def g(y):
			# parameters for the first truncated normal
			loc1 = 25
			scale1 = 6
			a1 = (m - loc1) / scale1
			b1 = (M - loc1) / scale1

			# parameters for the second truncated normal
			loc2 = 31
			scale2 = 6
			a2 = (m - loc2) / scale2
			b2 = (M - loc2) / scale2 

			return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

		xx = np.linspace(12, 37, 1000)
		gg = g(xx)
		mm = np.argmin(gg)
		MM = np.argmax(gg)
		gmin = g(xx[mm])
		gmax = g(xx[MM])

		if np.max(0.02 + 0.95 * (g(discounted_price) - gmin) / (gmax - gmin) <= 1):
			return 0.02 + 0.95 * (g(discounted_price) - gmin) / (gmax - gmin)
		else:
			return 0.95 * (g(discounted_price) - gmin) / (gmax - gmin)

    # Senior Amateur ########################################################################################### 
	if cl == 3:
		return np.exp(0.02 * (M - discounted_price)) / np.exp(0.02 * (M - m + 2))

CR2 = np.array([np.array([compute_cr2(discounted_m2, c) for c, _ in enumerate(NUM_CUSTOMERS) for discounted_m2 in m2]).reshape((len(NUM_CUSTOMERS), len(m2))) for m2 in MARGINS_2])

SD_CUSTOMERS = np.array([2, 4, 1, 3])  # standard deviation on the number of customers per each class

if SETTING == 0:
    PROMO_PROB = np.array([0.4, 0.2, 0.22, 0.18]) # Promo-allocation, fixed by the Business Unit of the shop
else:
    PROMO_PROB = np.array([0.2, 0.3, 0.1, 0.4]) # Promo-allocation, fixed by the Business Unit of the shop

def build_matrix(idx1, idx2): 
    tot_customers = np.sum(NUM_CUSTOMERS)

    n_promos = (PROMO_PROB[1 :] * tot_customers).astype(int)
    n_promos = np.insert(n_promos, 0, tot_customers - np.sum(n_promos))

    profit = CR1[idx1].reshape((4, 1)) * (MARGINS_1[idx1] + CR2[idx2] * MARGINS_2[idx2])

    # Repeat columns
    matrix = np.repeat(profit, n_promos, axis=1)

    # Repeat rows
    matrix = np.repeat(matrix, NUM_CUSTOMERS, axis=0)

    return matrix

def opt_matching(matrix):
    row, col = linear_sum_assignment(matrix, maximize=True)
    matching_mask = np.zeros(matrix.shape, dtype=int)
    matching_mask[row, col] = 1
    return matching_mask * matrix, matching_mask

def compute_opt_matching():
    opt_value = -1
    for a1 in range(N_ARMS_1):
        for a2 in range(N_ARMS_2):
            matrix = build_matrix(a1, a2)
            matching, _ = opt_matching(matrix) ##############################
            value = np.sum(matching)
            if value > opt_value:
                opt_value = value
                # idx1 = a1 ##############################
                # idx2 = a2 ##############################

    return opt_value  #, (idx1, idx2) ##############################

OPT = compute_opt_matching()