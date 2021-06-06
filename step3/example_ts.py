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

tmp0 = np.zeros(shape=config.NUM_CUSTOMERS[0], dtype=int)
tmp1 = np.ones(shape=config.NUM_CUSTOMERS[1], dtype=int)
tmp2 = np.zeros(shape=config.NUM_CUSTOMERS[2], dtype=int) + 2
tmp3 = np.zeros(shape=config.NUM_CUSTOMERS[3], dtype=int) + 3

customer_arrivals = np.array([], dtype=int)
customer_arrivals = np.concatenate((customer_arrivals, tmp0), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp1), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp2), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp3), axis=None)


for e in tqdm(range(config.N_EXPS)):
    env = Environment(n_arms=config.N_ARMS, cr1=config.CR1)
    ts_learner = TS_Learner(n_arms=config.N_ARMS)

    daily_rewards = []

    for t in range(config.T):
        np.random.shuffle(customer_arrivals)
        #for _ in range(tot_customers):
            #c_class = random.sample(customer_arrivals.toList(), 1)[0]
        daily_profits = 0
        for c_class in customer_arrivals:
            # Thompson Sampling
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm, c_class)  # questo deve diventare 0 o 1
            ts_learner.update(pulled_arm, reward, c_class)  # update solo della beta della classe del cliente corrente

            # reward * (margin1 + promo * margin2 * conv2[pulled_arm])
            avg_customer_profit = reward * (config.MARGINS_1[pulled_arm] + np.dot(np.multiply(config.MATCHING[c_class],
                                                                                              config.MARGINS_2),
                                                                                  config.CR2[c_class]) / config.NUM_CUSTOMERS[c_class])
            daily_profits += avg_customer_profit

        daily_rewards.append(daily_profits)

    ts_reward_per_experiment.append(daily_rewards)

print(np.shape(daily_rewards))
print(np.shape(ts_reward_per_experiment))

# Plot the results
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(config.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.show()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(config.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.show()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.plot(np.mean(config.OPT - ts_reward_per_experiment, axis=0), color='b')
plt.show()

#plt.plot(np.cumsum(np.mean(TOTAL_OPT - np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'g')  # Cumulative mean of regrets
#plt.plot(TOTAL_OPT - np.sum(ts_learner.collected_rewards, axis=1), color='blue')  # Daily regret
#plt.plot(np.cumsum(TOTAL_OPT - np.sum(ts_learner.collected_rewards, axis=1)), 'r')  # Cumulative daily regret

#plt.plot(np.cumsum(np.mean(TOTAL_OPT - np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'r')