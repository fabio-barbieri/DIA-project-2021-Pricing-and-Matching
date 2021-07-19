import json
import numpy as np
import matplotlib.pyplot as plt
from Environment_4 import *
from TS_Learner_4 import *
from tqdm import tqdm

np.random.seed(1234)

with open('setup/config.json') as config_file:
    config = json.load(config_file)
    config_file.close()

T = config['T']
N_EXPS = config['n_exps']
N_ARMS = config['n_arms']
NUM_CUSTOMERS = np.array(config['num_customers'])
MARGINS_1 = np.array(config['step4']['margins_1'])
CR1 = np.array(config['step4']['cr1'])
SD_CUSTOMERS = np.array(config['step4']['sd_customers'])
PROMO_PROB = np.array(config['step4']['promo_prob'])
MATCHING_PROB = np.array(config['step4']['matching_prob'])
MARGINS_2 = np.array(config['step4']['margins_2'])
CR2 = np.array(config['step4']['cr2'])
OPT = config['step4']['opt']

ts_reward_per_experiment = []  # Collected reward

for e in tqdm(range(N_EXPS)):
    env = Environment_4(n_arms=N_ARMS, 
                        cr1=CR1, 
                        cr2=CR2,
                        num_customers=NUM_CUSTOMERS,
                        sd_customers=SD_CUSTOMERS,
                        matching_prob=MATCHING_PROB)
    ts_learner = TS_Learner_4(n_arms=N_ARMS, 
                              tot_customers=np.sum(NUM_CUSTOMERS),
                              sd_customers=SD_CUSTOMERS,
                              margins_1=MARGINS_1,
                              matching_prob=MATCHING_PROB,
                              margins_2=MARGINS_2)

    daily_rewards = []
    for t in range(T):
        customer_arrivals, current_daily_customers = env.customers()
        daily_profits = 0
        for c_class in customer_arrivals:
            # Thompson Sampling
            pulled_arm = ts_learner.pull_arm()
            reward1, reward2, promo = env.round(pulled_arm, c_class)
            ts_learner.update(pulled_arm, reward1, reward2, c_class, promo)

            # reward1 * (margin1 + reward2 * margin2)
            customer_profit = reward1 * (MARGINS_1[pulled_arm] + reward2 * (MARGINS_2[promo]))
            daily_profits += customer_profit

        daily_rewards.append(daily_profits)

        ts_learner.compute_posterior(x_bar=current_daily_customers)

    ts_reward_per_experiment.append(daily_rewards)

ts_reward_per_experiment = np.array(ts_reward_per_experiment)

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.savefig(f"step4/plots/TS/TS_ExpRew_{N_EXPS}-{N_ARMS}.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.savefig(f"step4/plots/TS/TS_CumulativeExpRew_{N_EXPS}-{N_ARMS}.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.hlines(0, 0, 365, linestyles="dashed")
plt.plot(np.mean(OPT - ts_reward_per_experiment, axis=0), color='b')
plt.savefig(f"step4/plots/TS/TS_DailyRegret_{N_EXPS}-{N_ARMS}.png", dpi=200)
plt.show()
