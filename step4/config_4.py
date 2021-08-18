import numpy as np
from scipy.stats import truncnorm
import sys

SETTING = int(input('Specify the setting for the current experiment (0 or 1): '))
while (SETTING != 0) and (SETTING != 1):
    print('Wrong setting, try again!')
    SETTING = int(input('Specify the setting for the current experiment (0 or 1): '))

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

CR1 = np.array([compute_cr1(m1, c) for m1 in MARGINS_1 for c, _ in enumerate(NUM_CUSTOMERS)]).reshape(len(MARGINS_1), len(NUM_CUSTOMERS))

#                           p0  p1    p2    p3  
promo_discounts = np.array([1, 0.85, 0.75, 0.60])
MARGINS_2 = 29.99 * promo_discounts

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

CR2 = np.array(np.array([compute_cr2(discounted_m2, c) for c, _ in enumerate(NUM_CUSTOMERS) for discounted_m2 in MARGINS_2]).reshape((len(NUM_CUSTOMERS), len(MARGINS_2))))

SD_CUSTOMERS = np.array([2, 4, 1, 3])  # standard deviation on the number of customers per each class

if SETTING == 0:
    PROMO_PROB = np.array([0.4, 0.2, 0.22, 0.18]) # Promo-assignments for each class, fixed by the Business Unit of the shop
    
    # MATCHING_PROB[i,j] = Probability that a customer is of class i and gets promo j
    MATCHING_PROB = np.array([[0.08, 0.05, 0.04, 0.03], # Class 1
                              [0.16, 0.06, 0.10, 0.08], # Class 2
                              [0.02, 0.03, 0.03, 0.02], # Class 3
                              [0.14, 0.06, 0.05, 0.05]]) # Class 4
#                               p0    p1    p2    p3
else:
    PROMO_PROB = np.array([0.2, 0.3, 0.1, 0.4]) # Promo-assignments for each class, fixed by the Business Unit of the shop
    
    # MATCHING_PROB[i,j] = Probability that a customer is of class i and gets promo j
    MATCHING_PROB = np.array([[0.05, 0.07, 0.02, 0.06], # Class 1 -> tot = NUM_CUSTOMERS[0]
                         [0.05, 0.08, 0.01, 0.26], # Class 2 -> tot = NUM_CUSTOMERS[1]
                         [0.03, 0.03, 0.02, 0.02], # Class 3 -> tot = NUM_CUSTOMERS[2]
                         [0.07, 0.12, 0.05, 0.06]]) # Class 4 -> tot = NUM_CUSTOMERS[3]
#                         p0     p1    p2    p3

def compute_profit(i, margin1):
    tot_customers = np.sum(NUM_CUSTOMERS)
    matching_prob = MATCHING_PROB / np.expand_dims(NUM_CUSTOMERS, axis=1) * tot_customers
    a = CR1[i] * (margin1 + np.dot(CR2 * matching_prob, MARGINS_2))  #   4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                    # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                    # = 4x1 * (1x1 + 4x1) = 
                                                                    # = 4x1 * 4x1 = 
                                                                    # = 4x1
    return np.dot(a, NUM_CUSTOMERS)

known_profits = [compute_profit(i, m1) for i, m1 in enumerate(MARGINS_1)]
OPT = max(known_profits)
