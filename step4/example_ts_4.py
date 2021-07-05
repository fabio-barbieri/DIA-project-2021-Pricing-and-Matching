import config_4
import numpy as np
import matplotlib.pyplot as plt
from Environment_4 import *
from TS_Learner_4 import *
from tqdm import tqdm

np.random.seed(1234)

ts_reward_per_experiment = []  # Collected reward

#tot_customers = sum(config.NUM_CUSTOMERS)
#class_probabilities = [i / tot_customers for i in config.NUM_CUSTOMERS]


for e in tqdm(range(config_4.N_EXPS)):
    env = Environment_4(n_arms=config_4.N_ARMS, cr1=config_4.CR1, cr2=config_4.CR2)
    ts_learner = TS_Learner_4(n_arms=config_4.N_ARMS)

    daily_rewards = []

    for t in range(config_4.T):

        customer_arrivals, current_daily_customers = env.customers()

        daily_profits = 0
        for c_class in customer_arrivals:
            # Thompson Sampling
            pulled_arm = ts_learner.pull_arm()
            reward1, reward2, promo = env.round(pulled_arm, c_class, current_daily_customers)  # questo deve diventare 0 o 1
            ts_learner.update(pulled_arm, reward1, reward2, c_class, promo)  # update solo della beta della classe del cliente corrente

            # reward1 * (margin1 + reward2 * margin2)
            avg_customer_profit = reward1 * (config_4.MARGINS_1[pulled_arm] + reward2 * (config_4.MARGINS_2[promo]))
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
plt.hlines(config_4.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.savefig("expected_reward.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(config_4.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.savefig("cumulative_expected_reward.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.plot(np.mean(config_4.OPT - ts_reward_per_experiment, axis=0), color='b')
plt.savefig("daily_regret.png", dpi=200)
plt.show()
