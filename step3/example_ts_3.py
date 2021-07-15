import json
import numpy as np
import matplotlib.pyplot as plt
from Environment_3 import *
from TS_Learner_3 import *
from tqdm import tqdm

np.random.seed(1234)

with open('../setup/config.json') as config_file:
    config = json.load(config_file)
    config_file.close()

T = config['T']
N_EXPS = config['n_exps']
N_ARMS = config['n_arms']
NUM_CUSTOMERS = np.array(config['num_customers'])
MARGINS_1 = np.array(config['step3']['margins_1'])
CR1 = np.array(config['step3']['cr1'])
MATCHING = np.array(config['step3']['matching'])
MARGINS_2 = np.array(config['step3']['margins_2'])
CR2 = np.array(config['step3']['cr2'])
OPT = config['step3']['opt']

ts_reward_per_experiment = []  # Collected reward

tot_customers = np.sum(NUM_CUSTOMERS)
class_probabilities = [i / tot_customers for i in NUM_CUSTOMERS]

tmp0 = np.zeros(shape=NUM_CUSTOMERS[0], dtype=int)
tmp1 = np.ones(shape=NUM_CUSTOMERS[1], dtype=int)
tmp2 = np.zeros(shape=NUM_CUSTOMERS[2], dtype=int) + 2
tmp3 = np.zeros(shape=NUM_CUSTOMERS[3], dtype=int) + 3

customer_arrivals = np.array([], dtype=int)
customer_arrivals = np.concatenate((customer_arrivals, tmp0), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp1), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp2), axis=None)
customer_arrivals = np.concatenate((customer_arrivals, tmp3), axis=None)


for e in tqdm(range(N_EXPS)):
    env = Environment_3(n_arms=N_ARMS,
                        matching=MATCHING,
                        cr1=CR1,
                        cr2=CR2)
    ts_learner = TS_Learner_3(n_arms=N_ARMS, 
                              num_customers=NUM_CUSTOMERS, 
                              margins_1=MARGINS_1, 
                              matching=MATCHING, 
                              margins_2=MARGINS_2, 
                              cr2=CR2)

    daily_rewards = []
    for t in range(T):
        np.random.shuffle(customer_arrivals)
        daily_profits = 0
        for c_class in customer_arrivals:
            # Thompson Sampling
            pulled_arm = ts_learner.pull_arm()
            reward1, reward2, promo = env.round(pulled_arm, c_class)  # questo deve diventare 0 o 1
            ts_learner.update(pulled_arm, reward1, c_class)  # update solo della beta della classe del cliente corrente

            # reward * (margin1 + promo * margin2 * conv2[pulled_arm])
            customer_profit = reward1 * (MARGINS_1[pulled_arm] + reward2 * MARGINS_2[promo])
            daily_profits += customer_profit

        daily_rewards.append(daily_profits)

    ts_reward_per_experiment.append(daily_rewards)

ts_reward_per_experiment = np.array(ts_reward_per_experiment)

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected Reward")
plt.hlines(OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.savefig(f"plots/TS/TS_ExpRew_{N_EXPS}-{N_ARMS}.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Reward")
plt.hlines(OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.savefig(f"plots/TS/TS_CumulativeExpRew_{N_EXPS}-{N_ARMS}.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily Regret")
plt.hlines(0, 0, 365, linestyles="dashed")
plt.plot(np.mean(OPT - ts_reward_per_experiment, axis=0), color='b')
plt.savefig(f"plots/TS/TS_DailyRegret_{N_EXPS}-{N_ARMS}.png", dpi=200)
plt.show()