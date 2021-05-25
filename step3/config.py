import numpy as np

# Fix the seed for numpy in order to redo experiments
np.random.seed(1234)

T = 365  # Time horizon
N_EXPS = 200  # Number of experiments
N_ARMS = 5  # Number of different candidate prices
NUM_CUSTOMERS = np.array([20, 40, 10, 30])  # Mean of the number of total daily customers per class

# Computation of the Conv Rates of the first item for each class
CONV_RATES_1 = []
for _ in range(N_ARMS):
    CONV_RATES_1.append(np.random.uniform(0, 1, 4))  # Randomly sampled values from a distribution

# Computation of the weighted averages of the Conv Rates w.r.t. the number of customers per class
weighted_averages = []
for cr in CONV_RATES_1:
    weighted_averages.append(np.dot(cr, NUM_CUSTOMERS) / sum(NUM_CUSTOMERS))

OPT = CONV_RATES_1[np.argmax(weighted_averages)]  # Index of the highest weighted average of first Conv Rates

