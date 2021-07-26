import numpy as np
from scipy import stats
import sys

def cr1(season, price, cl):  # conversion rates for prices of item 1
 
  # MAXIMUM and minimun prices for item 1
  M = 250
  m = 150
  
  if price<m or price>M: 
    sys.exit('price not in range')

  
  if cl == 0:       # Junior Professional and (SPRING and AUTUMN)
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

    def f1(y): # Junior prof. SPRING and AUTUMN = cl == 0 and (season == 1 or season == 3)
      # parameters for the first truncated normal
      loc1 = 165
      scale1 = 50
      a1 = (m - loc1) / scale1
      b1 = (M - loc1) / scale1
      # parameters for the second truncated normal
      loc2 = 200
      scale2 = 80
      a2 = (m - loc2) / scale2
      b2 = (M - loc2) / scale2 

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(150,250,2000)
    ff = f1(xx)
    mm = np.argmin(ff)
    MM = np.argmax(ff)
    fmin1 = f1(xx[mm])
    fmax1 = f1(xx[MM])

    if season == 0:
      return 0.75 * (f(price) - fmin) / (fmax - fmin)
    elif season == 1 or season == 3:
      return 0.95 * (f(price) - fmin) / (fmax - fmin)
    else:
      return 0.95 * (f1(price) - fmin1) / (fmax1 - fmin1)


  if cl == 1:       # Junior Amateur and (SPRING or SUMMER)
    if season == 0:
      return np.exp(0.06*(M-price))/np.exp(0.06*(M-m+2)) * 0.8
    elif season == 1 or season == 2: 
      return np.exp(0.04*(M-price))/np.exp(0.04*(M-m+2))
    else: 
      return np.exp(0.04*(M-price))/np.exp(0.04*(M-m+2)) * 0.75


  if cl == 2:       # Senior Professional and (SPRING or AUTUMN)
    def g(y):
      # parameters for the first truncated normal
      loc1 = 200
      scale1 = 60
      a1 = (m - loc1) / scale1
      b1 = (M - loc1) / scale1
      # parameters for the second truncated normal
      loc2 = 230
      scale2 = 60
      a2 = (m - loc2) / scale2
      b2 = (M - loc2) / scale2 

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(150,250,2000)
    gg = g(xx)
    mm = np.argmin(gg)
    MM = np.argmax(gg)
    gmin = g(xx[mm])
    gmax = g(xx[MM])

    def g1(y):
      # parameters for the first truncated normal
      loc1 = 165
      scale1 = 60
      a1 = (m - loc1) / scale1
      b1 = (M - loc1) / scale1
      # parameters for the second truncated normal
      loc2 = 200
      scale2 = 60
      a2 = (m - loc2) / scale2
      b2 = (M - loc2) / scale2 

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(150,250,2000)
    gg = g1(xx)
    mm = np.argmin(gg)
    MM = np.argmax(gg)
    gmin1 = g1(xx[mm])
    gmax1 = g1(xx[MM])

    if season == 0:
      return 0.95 * (g1(price) - gmin1) / (gmax1 - gmin1)
    elif season == 1 or season == 3:
      return 0.95 * (g(price) - gmin) / (gmax - gmin)
    else:
      return 0.75 * (g(price) - gmin) / (gmax - gmin)


  if cl == 3:   # Senior Amateur and (SPRING and SUMMER)
    if season == 0:
      return np.exp(0.05*(M-price))/np.exp(0.05*(M-m+2)) * 0.8
    elif season == 1 or season == 2: 
      return np.exp(0.02*(M-price))/np.exp(0.02*(M-m+2))
    else: 
      return np.exp(0.02*(M-price))/np.exp(0.02*(M-m+2)) * 0.75


def cr2(season, price, cl):  # conversion rates for prices of item 1
 
  # MAXIMUM and minimun prices for item 1
  M = 37
  m = 12
  
  if price<m or price>M: 
    sys.exit('price not in range')

  
  if cl == 0:       # Junior Professional and (SPRING or AUTUMN)
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

    xx = np.linspace(12,37,1000)
    ff = f(xx)
    mm = np.argmin(ff)
    MM = np.argmax(ff)
    fmin = f(xx[mm])
    fmax = f(xx[MM])

    def f1(y):
      # parameters for the first truncated normal
      loc1 = 13
      scale1 = 5
      a1 = (m - loc1) / scale1
      b1 = (M - loc1) / scale1
      # parameters for the second truncated normal
      loc2 = 20
      scale2 = 8
      a2 = (m - loc2) / scale2
      b2 = (M - loc2) / scale2 

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(12,37,1000)
    ff = f1(xx)
    mm = np.argmin(ff)
    MM = np.argmax(ff)
    fmin1 = f1(xx[mm])
    fmax1 = f1(xx[MM])

    if season == 0:
      if np.max(0.02 + 0.75 * (f(price) - fmin) / (fmax - fmin) <= 1):
        return 0.02 + 0.75 * (f(price) - fmin) / (fmax - fmin)
      return 0.75 * (f(price) - fmin) / (fmax - fmin)
    elif season == 1 or season == 3:
      if np.max(0.02 + 0.95 * (f(price) - fmin) / (fmax - fmin) <= 1):
        return 0.02 + 0.95 * (f(price) - fmin) / (fmax - fmin)
      return 0.95 * (f(price) - fmin) / (fmax - fmin)
    else:
      if np.max(0.02 + 0.95 * (f1(price) - fmin1) / (fmax1 - fmin1) <= 1):
        return 0.02 + 0.95 * (f1(price) - fmin1) / (fmax1 - fmin1)
      return 0.95 * (f1(price) - fmin1) / (fmax1 - fmin1)


  if cl == 1:       # Junior Amateur and (SPRING or SUMMER)
    if season == 0:
      return np.exp(0.06*(M-price))/np.exp(0.06*(M-m+2)) * 0.8
    elif season == 1 or season == 2: 
      return np.exp(0.04*(M-price))/np.exp(0.04*(M-m+2))
    else: 
      return np.exp(0.04*(M-price))/np.exp(0.04*(M-m+2)) * 0.75    


  if cl == 2:       # Senior Professional and (SPRING or AUTUMN)
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

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(12,37,1000)
    gg = g(xx)
    mm = np.argmin(gg)
    MM = np.argmax(gg)
    gmin = g(xx[mm])
    gmax = g(xx[MM])

    def g1(y):
      # parameters for the first truncated normal
      loc1 = 13
      scale1 = 6
      a1 = (m - loc1) / scale1
      b1 = (M - loc1) / scale1
      # parameters for the second truncated normal
      loc2 = 20
      scale2 = 6
      a2 = (m - loc2) / scale2
      b2 = (M - loc2) / scale2 

      return stats.truncnorm.pdf(y,a1,b1,loc1,scale1)*stats.truncnorm.pdf(y,a2,b2,loc2,scale2)

    xx = np.linspace(12,37,1000)
    gg = g1(xx)
    mm = np.argmin(gg)
    MM = np.argmax(gg)
    gmin1 = g1(xx[mm])
    gmax1 = g1(xx[MM])

    if season == 0:
      if np.max(0.02 + 0.95 * (g1(price) - gmin1) / (gmax1 - gmin1) <= 1):
        return 0.02 + 0.95 * (g1(price) - gmin1) / (gmax1 - gmin1)
      return 0.95 * (g1(price) - gmin1) / (gmax1 - gmin1)
    elif season == 1 or season == 3:
      if np.max(0.02 + 0.95 * (g(price) - gmin) / (gmax - gmin) <= 1):
        return 0.02 + 0.95 * (g(price) - gmin) / (gmax - gmin)
      return 0.95 * (g(price) - gmin) / (gmax - gmin)
    else:
      if np.max(0.02 + 0.75 * (g(price) - gmin) / (gmax - gmin) <= 1):
        return 0.02 + 0.75 * (g(price) - gmin) / (gmax - gmin)
      return 0.75 * (g(price) - gmin) / (gmax - gmin)


  if cl == 3:   # Senior Amateur and (SPRING or SUMMER)
    if season == 0:
      return np.exp(0.05*(M-price))/np.exp(0.05*(M-m+2)) * 0.8
    elif season == 1 or season == 2: 
      return np.exp(0.02*(M-price))/np.exp(0.02*(M-m+2))
    else: 
      return np.exp(0.02*(M-price))/np.exp(0.02*(M-m+2)) * 0.75
