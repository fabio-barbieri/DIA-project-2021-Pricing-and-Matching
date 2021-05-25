import numpy as np
np.random.seed(1234)

n_arms = 5
num_customers = np.array([20, 40, 10, 30])  # mean of the number of total daily customers per class

first_conversion_rates = []
for _ in range(n_arms):
    first_conversion_rates.append(np.random.uniform(0, 1, 4))

weighted_averages = []
for cr in first_conversion_rates:
    weighted_averages.append(np.dot(cr, num_customers) / sum(num_customers))

opt = first_conversion_rates[np.argmax(weighted_averages)]

T = 365

n_exps = 200

