import numpy as np
from scipy import stats
import sys

def cr1(price,cl):  # conversion rates for prices of item 1

	# MAXIMUM and minimun prices for item 1
	M = 250
	m = 150

	if price < m or price > M:
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



def cr2(price,cl):  # conversion rates for prices of item 1

	# MAXIMUM and minimun prices for item 1
	M = 37
	m = 12

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

		xx = np.linspace(12,37,1000)
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

		if np.max(0.02 + 0.95 * (g(price) - gmin) / (gmax - gmin) <= 1):
			return 0.02 + 0.95 * (g(price) - gmin) / (gmax - gmin)
		else:
			return 0.95 * (g(price) - gmin) / (gmax - gmin)



	if cl == 3:       # Senior Amateur
		
		return np.exp(0.02*(M-price))/np.exp(0.02*(M-m+2))
