import config
import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *
from tqdm import tqdm
import random

np.random.seed(1234)

ts_reward_per_experiment = []  # Collected reward

tot_customers = sum(config.NUM_CUSTOMERS)
class_probabilities = [i / tot_customers for i in config.NUM_CUSTOMERS]


for e in tqdm(range(config.N_EXPS)):
    env = Environment(n_arms=config.N_ARMS, cr1=config.CR1, cr2=config.CR2)
    ts_learner = TS_Learner(n_arms=config.N_ARMS)

    daily_rewards = []

    for t in range(config.T):

        # extracting number of customer per class given a normal distribution
        tmp0 = np.zeros(shape=int(np.random.normal(config.NUM_CUSTOMERS[0], config.SD_CUSTOMERS[0])), dtype=int)
        tmp1 = np.ones(shape=int(np.random.normal(config.NUM_CUSTOMERS[1], config.SD_CUSTOMERS[1])), dtype=int)
        tmp2 = np.zeros(shape=int(np.random.normal(config.NUM_CUSTOMERS[2], config.SD_CUSTOMERS[2])), dtype=int) + 2
        tmp3 = np.zeros(shape=int(np.random.normal(config.NUM_CUSTOMERS[3], config.SD_CUSTOMERS[3])), dtype=int) + 3
        customer_arrivals = np.array([], dtype=int)
        customer_arrivals = np.concatenate((customer_arrivals, tmp0), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp1), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp2), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp3), axis=None)

        np.random.shuffle(customer_arrivals)

        current_daily_customers = np.array([len(tmp0), len(tmp1), len(tmp2), len(tmp3)])

        daily_profits = 0
        for c_class in customer_arrivals:
            # Thompson Sampling
            pulled_arm = ts_learner.pull_arm()
            reward1, reward2, promo = env.round(pulled_arm, c_class, current_daily_customers)  # questo deve diventare 0 o 1
            ts_learner.update(pulled_arm, reward1, reward2, c_class, promo)  # update solo della beta della classe del cliente corrente

            # reward1 * (margin1 + reward2 * margin2)
            avg_customer_profit = reward1 * (config.MARGINS_1[pulled_arm] + reward2 * (config.MARGINS_2[promo]))
            daily_profits += avg_customer_profit

        daily_rewards.append(daily_profits)

        #ts_learner.update_expected_customers(current_daily_customers, t + 1)
        ts_learner.compute_posterior(x_bar=current_daily_customers)

    ts_reward_per_experiment.append(daily_rewards)

print(np.shape(daily_rewards))
print(np.shape(ts_reward_per_experiment))

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(config.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.savefig("expected_reward.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(config.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.savefig("cumulative_expected_reward.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.plot(np.mean(config.OPT - ts_reward_per_experiment, axis=0), color='b')
plt.savefig("daily_regret.png", dpi=200)
plt.show()
