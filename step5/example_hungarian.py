import config_5
from Environment_5 import *
import numpy as np
from tqdm import tqdm
from Learner_5 import *
import matplotlib.pyplot as plt


np.random.seed(1234)

values_per_exp = []
#opt_per_exp = []
for e in tqdm(range(config_5.N_EXPS)):
    env = Environment_5(cr1=config_5.CR1, cr2=config_5.CR2)
    h_learner = Learner_5()

    daily_values = []
    for t in tqdm(range(config_5.T)):
        # Build the matching at the start of the day with data from t-1
        daily_values.append(np.sum(h_learner.compute_matching()))

        # Observe the actual number of arrived customers and their order
        customer_arrivals, current_daily_customers = env.customers()

        # Simulate the rewards and update the betas
        for c_class in customer_arrivals:
            reward1, reward2, promo = env.round(c_class)
            h_learner.update_betas(reward1, reward2, c_class, promo)

        # Compute posterior at the end of the day
        h_learner.compute_posterior(x_bar=current_daily_customers)

    values_per_exp.append(daily_values)
    #opt_per_exp.append(daily_values[-1])

# Plot the result
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(np.mean(config_5.OPT, axis=0) * config_5.T, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(values_per_exp, axis=0), axis=0), 'g')
plt.savefig("expected_values.png", dpi=200)
plt.show()

