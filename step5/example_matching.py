import config_5
from Environment_5 import *
import numpy as np
from tqdm import tqdm
from Learner_5 import *
import matplotlib.pyplot as plt

values_per_exp = []

for e in tqdm(range(config_5.N_EXPS)):
    env = Environment_5(cr1=config_5.CR1, 
                        cr2=config_5.CR2, 
                        num_customers=config_5.NUM_CUSTOMERS, 
                        sd_customers=config_5.SD_CUSTOMERS)
    h_learner = Learner_5(tot_customers=np.sum(config_5.NUM_CUSTOMERS), 
                          promo_prob=config_5.PROMO_PROB, 
                          margin_1=config_5.MARGIN_1, 
                          margins_2=config_5.MARGINS_2, 
                          sd_customers=config_5.SD_CUSTOMERS)

    daily_values = []
    for t in range(config_5.T):
        # Build the matching at the start of the day with data from t-1
        matching_values, matching_mask = h_learner.compute_matching()
        daily_values.append(np.sum(matching_values))

        # Compute matching_prob
        matching_prob = h_learner.compute_matching_prob(matching_mask)

        # Observe the actual number of arrived customers and their order
        customer_arrivals, current_daily_customers = env.customers()

        # Simulate the rewards and update the betas
        for c_class in customer_arrivals:
            reward1, reward2, promo = env.round(c_class, matching_prob, h_learner.expected_customers)
            h_learner.update_betas(reward1, reward2, c_class, promo)

        # Compute posterior at the end of the day
        h_learner.compute_posterior(x_bar=current_daily_customers)

    values_per_exp.append(daily_values)

# Plot the result
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected Profit")
plt.hlines(config_5.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(values_per_exp, axis=0), 'g')
plt.savefig(f"step5/plots/ExpProfit_{config_5.N_EXPS}e.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Profit")
plt.hlines(config_5.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(values_per_exp, axis=0), axis=0), 'r')
plt.savefig(f"step5/plots/CumulativeExpProfit_{config_5.N_EXPS}e.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.hlines(0, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(config_5.OPT - values_per_exp, axis=0)), color='c')
plt.savefig(f"step5/plots/CumulativeRegret_{config_5.N_EXPS}e.png", dpi=200)
plt.show()

