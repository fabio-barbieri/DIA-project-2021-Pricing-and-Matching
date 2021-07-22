import json
from Environment_5 import *
import numpy as np
from tqdm import tqdm
from Learner_5 import *
import matplotlib.pyplot as plt

with open('setup/config.json') as config_file:
    config = json.load(config_file)
    config_file.close()

T = config['T']
N_EXPS = config['n_exps']
NUM_CUSTOMERS = np.array(config['num_customers'])
MARGIN_1 = np.array(config['step5']['margin_1'])
CR1 = np.array(config['step5']['cr1'])
SD_CUSTOMERS = np.array(config['step5']['sd_customers'])
PROMO_PROB = np.array(config['step5']['promo_prob'])
MARGINS_2 = np.array(config['step5']['margins_2'])
CR2 = np.array(config['step5']['cr2'])
OPT = config['step5']['opt']

values_per_exp = []

for e in tqdm(range(N_EXPS)):
    env = Environment_5(cr1=CR1, 
                        cr2=CR2, 
                        num_customers=NUM_CUSTOMERS, 
                        sd_customers=SD_CUSTOMERS)
    h_learner = Learner_5(tot_customers=np.sum(NUM_CUSTOMERS), 
                          promo_prob=PROMO_PROB, 
                          margin_1=MARGIN_1, 
                          margins_2=MARGINS_2, 
                          sd_customers=SD_CUSTOMERS)

    daily_values = []
    for t in range(T):
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
plt.hlines(OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(values_per_exp, axis=0), 'g')
plt.savefig(f"step5/plots/ExpProfit_{N_EXPS}.png", dpi=200)
plt.show()

plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Profit")
plt.hlines(OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(values_per_exp, axis=0), axis=0), 'r')
plt.savefig(f"step5/plots/CumulativeExpProfit_{N_EXPS}.png", dpi=200)
plt.show()

