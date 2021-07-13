import numpy as np
from scipy import stats
import sys
from hungarian_algorithm import hungarian_algorithm
import config_6

def cr1(price,cl):  # conversion rates for prices of item 1
 
  # MAXIMUM and minimun prices for item 1
  M = 250
  m = 150
  
  if price<m or price>M: 
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

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(150,250,2000)
    ff = f(xx)
    mm = np.argmin(ff)
    MM = np.argmax(ff)
    fmin = f(xx[mm])
    fmax = f(xx[MM])

    return 0.95 * (f(price) - fmin) / (fmax - fmin)

  
  if cl == 1:       # Junior Amateur
   
      return np.exp(0.04*(M-price))/np.exp(0.04*(M-m+2))
    

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

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(150,250,2000)
    gg = g(xx)
    mm = np.argmin(gg)
    MM = np.argmax(gg)
    gmin = g(xx[mm])
    gmax = g(xx[MM])

    return 0.95 * (g(price) - gmin) / (gmax - gmin)


  if cl == 3:       # Senior Amateur
    
    return np.exp(0.02*(M-price))/np.exp(0.02*(M-m+2))



def cr2(price,cl):  # conversion rates for prices of item 2
 
  # MAXIMUM and minimun prices for item 2
  M = 35.90
  m = 15.90
  
  if price<m or price>M: 
    sys.exit('price not in range')

  
  if cl == 0:       # Junior Professional
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

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(15.90,35.90,1000)
    ff = f(xx)
    mm = np.argmin(ff)
    MM = np.argmax(ff)
    fmin = f(xx[mm])
    fmax = f(xx[MM])

    return 0.95 * (f(price) - fmin) / (fmax - fmin)

  
  if cl == 1:       # Junior Amateur
   
      return np.exp(0.04*(M-price))/np.exp(0.04*(M-m+2))
    

  if cl == 2:       # Senior Professional
    
    def g(y):
      # parameters for the first truncated normal
      loc1 = 25
      scale1 = 6
      a1 = (m - loc1) / scale1
      b1 = (M - loc1) / scale1
      # parameters for the second truncated normal
      loc2 = 31
      scale2 = 4
      a2 = (m - loc2) / scale2
      b2 = (M - loc2) / scale2 

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(15.90,35.90,1000)
    gg = g(xx)
    mm = np.argmin(gg)
    MM = np.argmax(gg)
    gmin = g(xx[mm])
    gmax = g(xx[MM])

    return 0.95 * (g(price) - gmin) / (gmax - gmin)


  if cl == 3:       # Senior Amateur
    
    return np.exp(0.02*(M-price))/np.exp(0.02*(M-m+2))
   


def build_optimal_matching(num_customers, promo_prob, conv1, conv2, margin_1, margins_2):
    matrix_dim = np.sum(num_customers)
    matrix = np.array([])

    # array containing the number of promos, assigned by the marketing unit
    n_promos = np.array([int(promo_prob[i] * matrix_dim) for i in range(1, 4)])
    n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))

    customer0_row = np.array([])
    customer1_row = np.array([])
    customer2_row = np.array([])
    customer3_row = np.array([])

    for i, n_promo in enumerate(n_promos):
        customer0_row = np.append(customer0_row, [conv1[0] * (margin_1 + margins_2[i] * conv2[0, i]) for _ in range(n_promo)])
        customer1_row = np.append(customer1_row, [conv1[1] * (margin_1 + margins_2[i] * conv2[1, i]) for _ in range(n_promo)])
        customer2_row = np.append(customer2_row, [conv1[2] * (margin_1 + margins_2[i] * conv2[2, i]) for _ in range(n_promo)])
        customer3_row = np.append(customer3_row, [conv1[3] * (margin_1 + margins_2[i] * conv2[3, i]) for _ in range(n_promo)])

    for i, num in enumerate(num_customers):
        for _ in range(num):
            if i == 0:
                matrix = np.concatenate((matrix, customer0_row), axis=0)
            elif i == 1:
                matrix = np.concatenate((matrix, customer1_row), axis=0)
            elif i == 2:
                matrix = np.concatenate((matrix, customer2_row), axis=0)
            else:
                matrix = np.concatenate((matrix, customer3_row), axis=0)

    matrix = np.reshape(matrix, (matrix_dim, matrix_dim))

    # computing optimal value for plotting purposes
    optimal_value = np.sum(hungarian_algorithm(matrix))

    return optimal_value