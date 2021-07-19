import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from hungarian_algorithm import hungarian_algorithm

# Conversion Rates for prices of item_1 ------------------------------------------------------------------------
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

            return stats.truncnorm.pdf(y, a1, b1, loc1, scale1) * stats.truncnorm.pdf(y, a2, b2, loc2, scale2)

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

            return stats.truncnorm.pdf(y, a1, b1, loc1, scale1) * stats.truncnorm.pdf(y, a2, b2, loc2, scale2)

        xx = np.linspace(150,250,2000)
        gg = g(xx)
        mm = np.argmin(gg)
        MM = np.argmax(gg)
        gmin = g(xx[mm])
        gmax = g(xx[MM])

        return 0.95 * (g(price) - gmin) / (gmax - gmin)

    # Senior Amateur ########################################################################################### 
    if cl == 3:
        return np.exp(0.02*(M-price))/np.exp(0.02*(M-m+2))

# Opt profit for step 3 and 4 ----------------------------------------------------------------------------------
def profit(i, cr1, margin1, cr2, margins2, matching, num_customers, step):
    tot_customers = np.sum(num_customers) if step == 4 else 1

    matching_prob = matching / np.expand_dims(num_customers, axis=1) * tot_customers
    a = cr1[i] * (margin1 + np.dot(cr2 * matching_prob, margins2))  # 4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                    # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                    # = 4x1 * (1x1 + 4x1) = 
                                                                    # = 4x1 * 4x1 = 
                                                                    # = 4x1
    return np.dot(a, num_customers)

# Matrix building for step 5 opt
def build_matrix(num_customers, promo_prob, cr1, cr2, margin_1, margins_2):
    matrix_dim = np.sum(num_customers)

    # First set integers p1, p2, p3 and the remaining are p0 
    n_promos = (promo_prob[1 :] * matrix_dim).astype(int)
    n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))

    # profit = cr1.reshape((4, 1)) * (margin_1 + cr2 * margins_2) ##################################################
    profit = cr1 * (margin_1 + cr2 * margins_2)                   ##################################################

    # repeat columns
    matrix = np.repeat(profit, n_promos, axis=1)

    # repeat rows
    matrix = np.repeat(matrix, num_customers, axis=0)

    return matrix

# Opt matching for step 5 --------------------------------------------------------------------------------------
def opt_matching(num_customers, promo_prob, cr1, cr2, margin_1, margins_2):
    matrix = build_matrix(num_customers, promo_prob, cr1, cr2, margin_1, margins_2)
    optimal_value = np.sum(hungarian_algorithm(matrix)[0])

    return optimal_value

if __name__ == '__main__':
    import json
    
    config = {}

    # COMMON FOR ALL STEPS ----------------------------------------------------
    T = 365
    config['T'] = T

    n_exps = 30
    config['n_exps'] = n_exps

    n_arms = 10
    config['n_arms'] = n_arms

    num_customers = np.array([20, 40, 10, 30])
    config['num_customers'] = num_customers.tolist()


    # STEP 1 ------------------------------------------------------------------
    #TODO


    # STEP 2 ------------------------------------------------------------------
    # /
    
    # STEP 3 ------------------------------------------------------------------
    step3 = {}

    margins_1 = np.linspace(150, 250, n_arms)
    step3['margins_1'] = margins_1.tolist()
    
    cr1 = []
    for margin in margins_1:
        cr = np.array([compute_cr1(margin, c_class) for c_class in range(len(num_customers))])
        cr1.append(cr)
    cr1_list = [el.tolist() for el in cr1]
    step3['cr1'] = cr1_list

    matching = np.array([[ 8,  5,  4,  3],  # Class 1 (Junior Professionals)
                         [16,  6, 10,  8],  # Class 2 (Junior Amateur)
                         [ 2,  3,  3,  2],  # Class 3 (Senior Professionals)
                         [14,  6,  5,  5]]) # Class 4 (Senior Amateur)
                        # p0   p1  p2  p3
    step3['matching'] = matching.tolist()

    margins_2 = np.array([29.99, 24.99, 20.99, 10.99])
    step3['margins_2'] = margins_2.tolist()

    cr2 = np.array([[0.2, 0.4, 0.3, 0.3],  # Class 1 (Junior Professionals)
                    [0.0, 0.2, 0.3, 0.5],  # Class 2 (Junior Amateur)
                    [0.1, 0.5, 0.3, 0.1],  # Class 3 (Senior Professionals)
                    [0.1, 0.1, 0.1, 0.7]]) # Class 4 (Senior Amateur)
                    # p0   p1   p2   p3
    step3['cr2'] = cr2.tolist()

    known_profits = np.array([profit(i, cr1, m1, cr2, margins_2, matching, num_customers, step=3) for i, m1 in enumerate(margins_1)])
    opt3 = np.max(known_profits)
    step3['opt'] = opt3

    config['step3'] = step3
    

    # STEP 4 ------------------------------------------------------------------
    step4 = {}

    sd_customers = np.array([2, 4, 1, 3])
    step4['sd_customers'] = sd_customers.tolist()

    step4['margins_1'] = margins_1.tolist()

    step4['cr1'] = cr1_list

    promo_prob = np.array([0.4, 0.2, 0.22, 0.18])
    step4['promo_prob'] = promo_prob.tolist()

    matching_prob = np.array([[0.08, 0.05, 0.04, 0.03],  # Class 1 (Junior Professionals)
                    	      [0.16, 0.06, 0.10, 0.08],  # Class 2 (Junior Amateur) 
                     	      [0.02, 0.03, 0.03, 0.02],  # Class 3 (Senior Professionals)
                     	      [0.14, 0.06, 0.05, 0.05]]) # Class 4 (Senior Amateur) 
                            #   p0     p1    p2    p3
    step4['matching_prob'] = matching_prob.tolist()

    step4['margins_2'] = margins_2.tolist()

    step4['cr2'] = cr2.tolist()

    known_profits = np.array([profit(i, cr1, m1, cr2, margins_2, matching_prob, num_customers, step=4) for i, m1 in enumerate(margins_1)])
    opt4 = np.max(known_profits)
    step4['opt'] = opt4
    
    config['step4'] = step4
    
    
    # STEP 5 ------------------------------------------------------------------
    step5 = {}

    margin_1 = 200
    step5['margin_1'] = margin_1

    cr1 = np.array([compute_cr1(margin_1, c_class) for c_class in range(len(num_customers))])
    step5['cr1'] = cr1.tolist()

    step5['sd_customers'] = sd_customers.tolist()
    
    step5['promo_prob'] = promo_prob.tolist()

    step5['margins_2'] = margins_2.tolist()

    step5['cr2'] = cr2.tolist()

    opt5 = opt_matching(num_customers, promo_prob, cr1, cr2, margin_1, margins_2)
    step5['opt'] = opt5

    config['step5'] = step5
    
    
    # STEP 6 ------------------------------------------------------------------
    step6 = {}
    #TODO
    
    
    # STEP 7 ------------------------------------------------------------------
    step7 = {}
    #TODO
    
    
    # STEP 8 ------------------------------------------------------------------
    step8 = {}
    #TODO

    with open('setup/config.json', 'w') as config_file:
        json.dump(config, config_file, indent=4)
        config_file.close()


