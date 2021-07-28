import config_3
import numpy as np
import matplotlib.pyplot as plt
from Environment_3 import *
from UCB1_Learner_3 import *
from tqdm import tqdm

np.random.seed(1234)

ucb1_reward_per_experiment = []  # Collected reward

tot_customers = sum(config_3.NUM_CUSTOMERS)
class_probabilities = [i / tot_customers for i in config_3.NUM_CUSTOMERS]

tmp0 = np.zeros(shape=config_3.NUM_CUSTOMERS[0], dtype=int)
tmp1 = np.ones(shape=config_3.NUM_CUSTOMERS[1], dtype=int)
tmp2 = np.zeros(shape=config_3.NUM_CUSTOMERS[2], dtype=int) + 2
tmp3 = np.zeros(shape=config_3.NUM_CUSTOMERS[3], dtype=int) + 3
customer_arrivals = np.concatenate((tmp0, tmp1, tmp2, tmp3), axis=None)

np.random.shuffle(customer_arrivals)

for e in tqdm(range(config_3.N_EXPS)):
    env = Environment_3(n_arms=config_3.N_ARMS,
                        matching=config_3.MATCHING,
                        cr1=config_3.CR1,
                        cr2=config_3.CR2)
    ucb1_learner = UCB1_Learner_3(n_arms=config_3.N_ARMS, 
                                  num_customers=config_3.NUM_CUSTOMERS, 
                                  margins_1=config_3.MARGINS_1, 
                                  matching=config_3.MATCHING, 
                                  margins_2=config_3.MARGINS_2, 
                                  cr2=config_3.CR2)

    daily_rewards = []
    for t in range(config_3.T):
        np.random.shuffle(customer_arrivals)
        daily_profits = 0
        for c_class in customer_arrivals:
            # UCB1
            pulled_arm = ucb1_learner.pull_arm()
            reward1, reward2, promo = env.round(pulled_arm, c_class)
            ucb1_learner.update(pulled_arm, reward1, reward2, promo)
            customer_profit = reward1 * (config_3.MARGINS_1[pulled_arm] + reward2 * config_3.MARGINS_2[promo])
            daily_profits += customer_profit

        daily_rewards.append(daily_profits)

    ucb1_reward_per_experiment.append(daily_rewards)

ucb1_reward_per_experiment = np.array(ucb1_reward_per_experiment)

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected Reward")
plt.hlines(config_3.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ucb1_reward_per_experiment, axis=0), 'g')
plt.savefig(f"step3/plots/UCB/UCB_ExpRew_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Reward")
plt.hlines(config_3.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ucb1_reward_per_experiment, axis=0)), 'r')
plt.savefig(f"step3/plots/UCB/UCB_CumulativeExpRew_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily Regret")
plt.hlines(0, 0, 365, linestyles="dashed")
plt.plot(np.mean(config_3.OPT - ucb1_reward_per_experiment, axis=0), color='b')
plt.savefig(f"step3/plots/UCB/UCB_DailyRegret_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.hlines(0, 0, 365, linestyles="dashed")
plt.plot(np.sumsum(np.mean(config_3.OPT - ucb1_reward_per_experiment, axis=0)), color='c')
plt.savefig(f"step3/plots/UCB/UCB_CumulativeRegret_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
plt.show()
