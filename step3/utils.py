import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

def cr1(price,cl):
  # MAXIMUM and minimun prices for item 1
  M = 250
  m = 150
  
  if price<m or price>M: 
    sys.exit('price not in range')

  if cl == 0:
    # Junior Professional
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

    return (f(price) - fmin) / (fmax - fmin)

  if cl == 1:
    # Junior Amateur
   
    ret = np.exp(0.04*(M-price))
    return ret / np.exp(0.04*(M-m))
    

  #if cl == 2:
    # Senior Professional
    

  #if cl == 3:
    # Senior Amateur