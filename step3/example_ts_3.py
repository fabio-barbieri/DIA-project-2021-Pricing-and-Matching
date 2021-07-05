import config_3
import numpy as np
import matplotlib.pyplot as plt
from Environment_3 import *
from TS_Learner_3 import *
from tqdm import tqdm

np.random.seed(1234)

ts_reward_per_experiment = []  # Collected reward

tot_customers = sum(config_3.NUM_CUSTOMERS)
class_probabilities = [i / tot_customers for i in config_3.NUM_CUSTOMERS]

tmp0 = np.zeros(shape=config_3.NUM_CUSTOMERS[0], dtype=int)
tmp1 = np.ones(shape=config_3.NUM_CUSTOMERS[1], dtype=int)
tmp2 = np.zeros(shape=config_3.NUM_CUSTOMERS[2], dtype=int) + 2
tmp3 = np.zeros(shape=config_3.NUM_CUSTOMERS[3], dtype=int) + 3

customer_arrivals = np.array([], dtype=int)
customer_arrivals = np.concatenate((customer_arrivals, tmp0), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp1), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp2), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp3), axis=None)


for e in tqdm(range(config_3.N_EXPS)):
    env = Environment_3(n_arms=config_3.N_ARMS, cr1=config_3.CR1)
    ts_learner = TS_Learner_3(n_arms=config_3.N_ARMS)

    daily_rewards = []

    for t in range(config_3.T):
        np.random.shuffle(customer_arrivals)
        daily_profits = 0
        for c_class in customer_arrivals:
            # Thompson Sampling
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm, c_class)  # questo deve diventare 0 o 1
            ts_learner.update(pulled_arm, reward, c_class)  # update solo della beta della classe del cliente corrente

            # reward * (margin1 + promo * margin2 * conv2[pulled_arm])
            avg_customer_profit = reward * (config_3.MARGINS_1[pulled_arm] + np.dot(np.multiply(config_3.MATCHING[c_class],
                                                                                                config_3.MARGINS_2),
                                                                                    config_3.CR2[c_class]) / config_3.NUM_CUSTOMERS[c_class])
            daily_profits += avg_customer_profit

        daily_rewards.append(daily_profits)

    ts_reward_per_experiment.append(daily_rewards)

print(np.shape(daily_rewards))
print(np.shape(ts_reward_per_experiment))

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(config_3.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.savefig("expected_reward.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(config_3.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.savefig("cumulative_expected_reward.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.plot(np.mean(config_3.OPT - ts_reward_per_experiment, axis=0), color='b')
plt.savefig("daily_regret.png", dpi=200)
plt.show()