# All CAPITAL-LETTERS variables are meant to have a fixed value

import pulp
import random
from operator import itemgetter

random.seed(1234) # In order to repeat always the same experiment

P1_VALUES = [1, 2, 3, 4, 5] # To be properly set -----------------------
P2_VALUES = [6, 7, 8, 9, 10] # To be properly set ----------------------

CUSTOMERS_PER_CLASS = [2, 3, 3, 2] # To be properly set ----------------
TOT_C = 0
for val in CUSTOMERS_PER_CLASS:
    TOT_C += val

PROMOS_PER_CLASS = [2, 4, 2, 2] # To be properly set -------------------
TOT_P = 0
for val in PROMOS_PER_CLASS:
    TOT_P += val
    
PROMOS = []
for _ in range(PROMOS_PER_CLASS[0]):
    PROMOS.append("P0")
for _ in range(PROMOS_PER_CLASS[1]):
    PROMOS.append("P1")
for _ in range(PROMOS_PER_CLASS[2]):
    PROMOS.append("P2")
for _ in range(PROMOS_PER_CLASS[3]):
    PROMOS.append("P3")    

PROMOS_VALUES = {"P0": 0.0,
                 "P1": 0.1,
                 "P2": 0.3,
                 "P3": 0.5} # To be properly set ------------------------

CONV_RATE1 = {}
for p1 in P1_VALUES:
    CONV_RATE1[str(p1)] = [random.random() for _ in range(4)]

CONV_RATE2 = {}
for p2 in P2_VALUES:
    discounted_p2 = {}
    for promo in PROMOS:
        discounted_p2[str(p2 * (1 - PROMOS_VALUES[promo]))] = [random.random() for _ in range(4)]
    CONV_RATE2[str(p2)] = discounted_p2
    
COST1 = 0 # To be properly set -------------------------------------------
COST2 = 0 # To be properly set -------------------------------------------



# ----------------------------------------------------------------    
# Fuctions -------------------------------------------------------
# ----------------------------------------------------------------    

def profit1(p1):
    
    """
    Input: A value for the price of Item_1 
    
    Returns: The profit associated with the input w.r.t the conversion rate
             for Item_1
    """
    
    margin = p1 - COST1
    cr = CONV_RATE1[str(p1)]
    s = 0
    for i in range(4):
        tmp = cr[i] * CUSTOMERS_PER_CLASS[i]
        s += tmp
    
    return margin * s
    


def customer2class(customer_idx):
    
    """
    Input: An index corresponding to a customer and 
    
    Returns: The class of the given customer (w.r.t. the number of 
             customers per class set at the beginning)
    """
    
    c1_idx = CUSTOMERS_PER_CLASS[0] - 1
    c2_idx = c1_idx + CUSTOMERS_PER_CLASS[1]
    c3_idx = c2_idx + CUSTOMERS_PER_CLASS[2]
    
    if customer_idx <= c1_idx:
        return 0
    
    if customer_idx > c1_idx and customer_idx <= c2_idx:
        return 1
    
    if customer_idx > c2_idx and customer_idx <= c3_idx:
        return 2
    
    if customer_idx > c3_idx:
        return 3



def profit2(p1, p2):
    
    """
    Input: A value for the price of Item_1 a value for the price of Item_2 
    
    Returns: The profit associated with the inputs w.r.t the conversion rate
             of Item_1 and Item_2, considering also the optimal matching 
             (which is found solving a LP problem)
    """
    
    # Problem
    prob = pulp.LpProblem("lp", pulp.LpMaximize)

    # Variables
    x = pulp.LpVariable.dicts(name="x", indexs=[(i,j) for i in range(TOT_C) 
                                                      for j in range(TOT_P)], lowBound=0, upBound=1, cat='Binary') 
    
    # Objective Function 
    prob += pulp.lpSum([(p2 * (1-PROMOS_VALUES[PROMOS[j]]) - COST2) *
                        CONV_RATE1[str(p1)][customer2class(i)] *
                        CONV_RATE2[str(p2)][str(p2 * (1-PROMOS_VALUES[PROMOS[j]]))][customer2class(i)] *
                        x[(i,j)]
                        for i in range(TOT_C) for j in range(TOT_P)]) 
    
    # Constraints 
    for i in range(TOT_C): 
        prob += pulp.lpSum(x[(i,j)] for j in range(TOT_P)) <= 1.0
        
    for j in range(TOT_P): 
        prob += pulp.lpSum(x[(i,j)] for i in range(TOT_C)) <= 1.0
        
    # Solution 
    prob.solve() 
    
    # Matching
    matching = [(i, j) for i in range(TOT_C) for j in range(TOT_P) if x[(i,j)].varValue == 1.0]
    
    return pulp.value(prob.objective), matching

# ----------------------------------------------------------------    
# ----------------------------------------------------------------  
# ----------------------------------------------------------------    
  
  

if TOT_C != TOT_P:
    print("----- ERROR: Too much customers or too much promos -----")
else:    
    results = []
    for p1 in P1_VALUES:
        for p2 in P2_VALUES:
            curr_first_profit = profit1(p1)
            curr_second_profit, curr_matching = profit2(p1, p2)
            curr_tot_profit = curr_first_profit + curr_second_profit
            results.append((p1, p2, curr_tot_profit, curr_matching))
            
    # Get the tuple (p1, p2, tot_profit, matching) whose tot_profit is the maximum     
    optimal_pricing_matching = max(results, key=itemgetter(2))
    
    optimal_p1 = optimal_pricing_matching[0]
    optimal_p2 = optimal_pricing_matching[1]
    optimal_total_profit = optimal_pricing_matching[2]
    optimal_matching = optimal_pricing_matching[3]
    
    print("Optimal Price for Item_1: {:.2f}".format(optimal_p1))
    print()
    print("Optimal Price for Item_2: {:.2f}".format(optimal_p2))
    print()
    print("Optimal Total Profit: {:.2f}".format(optimal_total_profit))
    print()
    print("Optimal matching:")
    for el in optimal_matching:
        print("{} --> {}".format(el[0], el[1]))
    
