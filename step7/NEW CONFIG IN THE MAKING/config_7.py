import numpy as np
from scipy.stats import truncnorm
import sys
from scipy.optimize import linear_sum_assignment

T = 365  # Time horizon

N_EXPS = 1  # Number of experiments

N_ARMS_1 = 5  # Number of different candidate prices

N_ARMS_2 = 5  # Number of candidates for second item price

NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class

SEASONS = ('Winter', 'Spring', 'Summer', 'Autumn') # Seasons of the year

MARGINS_1 = np.linspace(150, 250, N_ARMS_1)

def compute_cr1(season, price, cl):
    # MAXIMUM and minimun prices for item 1
    M = 250
    m = 150
  
    if price < m or price > M: 
        sys.exit('Price not in range')

    # Junior Professional (Best seasons: SPRING and AUTUMN) ######################################################################################    
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

        def f1(y):
            # Parameters for the first truncated normal
            loc1 = 165
            scale1 = 50
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1

            # Parameters for the second truncated normal
            loc2 = 200
            scale2 = 80
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 

            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(150, 250, 2000)
        ff = f1(xx)
        mm = np.argmin(ff)
        MM = np.argmax(ff)
        fmin1 = f1(xx[mm])
        fmax1 = f1(xx[MM])

        if season == 'Winter':
            return 0.75 * (f(price) - fmin) / (fmax - fmin)
        elif (season == 'Spring') or (season == 'Autumn'):
            return 0.95 * (f(price) - fmin) / (fmax - fmin)
        else:
            return 0.95 * (f1(price) - fmin1) / (fmax1 - fmin1)

    # Junior Amateur (Best seasons: SPRING and SUMMER)###########################################################################################
    if cl == 1:
        if season == 'Winter':
            return np.exp(0.06*(M-price))/np.exp(0.06*(M-m+2)) * 0.8
        elif (season == 'Spring') or (season == 'Summer'): 
            return np.exp(0.04*(M-price))/np.exp(0.04*(M-m+2))
        else: 
            return np.exp(0.04*(M-price))/np.exp(0.04*(M-m+2)) * 0.75

    # Senior Professional (Best seasons: SPRING and AUTUMN) ###########################################################################################
    if cl == 2:
        def g(y):
            # Parameters for the first truncated normal
            loc1 = 200
            scale1 = 60
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1

            # Parameters for the second truncated normal
            loc2 = 230
            scale2 = 60
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 

            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(150, 250, 2000)
        gg = g(xx)
        mm = np.argmin(gg)
        MM = np.argmax(gg)
        gmin = g(xx[mm])
        gmax = g(xx[MM])

        def g1(y):
            # Parameters for the first truncated normal
            loc1 = 165
            scale1 = 60
            a1 = (m - loc1) / scale1
            b1 = (M - loc1) / scale1
            
            # Parameters for the second truncated normal
            loc2 = 200
            scale2 = 60
            a2 = (m - loc2) / scale2
            b2 = (M - loc2) / scale2 

            return truncnorm.pdf(y, a1, b1, loc1, scale1) * truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(150, 250, 2000)
        gg = g1(xx)
        mm = np.argmin(gg)
        MM = np.argmax(gg)
        gmin1 = g1(xx[mm])
        gmax1 = g1(xx[MM])

        if season == 'Winter':
            return 0.95 * (g1(price) - gmin1) / (gmax1 - gmin1)
        elif season == 'Spring' or season == 'Autumn':
            return 0.95 * (g(price) - gmin) / (gmax - gmin)
        else:
            return 0.75 * (g(price) - gmin) / (gmax - gmin)

    # Senior Amateur (Best seasons: SPRING and SUMMER) ########################################################################################### 
    if cl == 3:
        if season == 'Winter':
            return np.exp(0.05 * (M - price)) / np.exp(0.05 * (M - m + 2)) * 0.8
        elif season == 'Spring' or season == 'Summer': 
            return np.exp(0.02 * (M - price)) / np.exp(0.02 * (M - m + 2))
        else: 
            return np.exp(0.02 * (M - price)) / np.exp(0.02 *(M - m + 2)) * 0.75

CR1 = np.array([[np.array([compute_cr1(season, m1, c)]) for m1 in MARGINS_1 for c, _ in enumerate(NUM_CUSTOMERS)] for season in SEASONS])
#######################
CR1 = []
# constructing matrix of conversion rates for the first product
for season in range(SEASONS):
    tmp = []
    for margin in MARGINS_1:
        cr = np.array([compute_cr1(season, margin, c_class) for c_class in range(len(NUM_CUSTOMERS))])
        tmp.append(cr)
    CR1.append(tmp)
CR1 = np.array(CR1)
######################

###
# TODO: CONTINUARE DA QUI E CONTROLLARE CR1 E IL SUO RESHAPE
###

PROMO_DISCOUNTS = np.array([1, 0.85, 0.75, 0.60])

MARGINS_2 = np.linspace(25, 35, N_ARMS_2).reshape((N_ARMS_2, 1)) * PROMO_DISCOUNTS.reshape((1, 4))

CR2 = []
# constructing matrix of conversion rates for the second product
for season in range(N_PHASES):
    tmp2 = []
    for margin in MARGINS_2:
        tmp1 = []
        for c_class in range(len(NUM_CUSTOMERS)):
            cr = np.array([utils_7.cr2(season, discounted_margin, c_class) for discounted_margin in margin])
            tmp1.append(cr)
        tmp2.append(tmp1)
    CR2.append(tmp2)

CR2 = np.array(CR2)

SD_CUSTOMERS = np.array([2, 4, 1, 3])  # standard deviation on the number of customers per each class

PROMO_PROB = np.array([0.4, 0.2, 0.22, 0.18]) # Promo-assignments for each class, fixed by the Business Unit of the shop

WINDOW_SIZE = int(np.sqrt(T))

def compute_opt_matching(season):
        opt_value = -1
        for arm_1 in range(N_ARMS_1):  # For every price_1
            for arm_2 in range(N_ARMS_2):
                matching, _ = hungarian_algorithm(build_matrix(season, arm_1, arm_2))
                value = np.sum(matching)
                if value > opt_value:
                    opt_value = value
                    # idx1 = arm_1
                    # idx2 = arm_2

        return opt_value  #, (idx1, idx2)

def build_matrix(season, idx1, idx2): 
        n_promos = (PROMO_PROB[1 :] * TOT_CUSTOMERS).astype(int)
        n_promos = np.insert(n_promos, 0, TOT_CUSTOMERS - np.sum(n_promos))

        profit = CR1[season][idx1].reshape((4, 1)) * (MARGINS_1[idx1] + CR2[season][idx2] * MARGINS_2[idx2])

        # repeat columns
        matrix = np.repeat(profit, n_promos, axis=1)

        # repeat rows
        matrix = np.repeat(matrix, NUM_CUSTOMERS, axis=0)

        return matrix

OPT = []
for season in range(N_PHASES):
    OPT.append(compute_opt_matching(season))

OPT = np.array(OPT)

if __name__ == '__main__':
    print()
    print()
    print()
