import config_4
import numpy as np
import matplotlib.pyplot as plt
from Environment_4 import *
from TS_Learner_4 import *
from tqdm import tqdm

np.random.seed(1234)


ts_reward_per_experiment = []  # Collected reward

for e in tqdm(range(config_4.N_EXPS)):
    env = Environment_4(n_arms=config_4.N_ARMS, 
                        cr1=config_4.CR1, 
                        cr2=config_4.CR2,
                        num_customers=config_4.NUM_CUSTOMERS,
                        sd_customers=config_4.SD_CUSTOMERS,
                        matching_prob=config_4.MATCHING_PROB)
    ts_learner = TS_Learner_4(n_arms=config_4.N_ARMS, 
                              tot_customers=np.sum(config_4.NUM_CUSTOMERS),
                              sd_customers=config_4.SD_CUSTOMERS,
                              margins_1=config_4.MARGINS_1,
                              matching_prob=config_4.MATCHING_PROB,
                              margins_2=config_4.MARGINS_2)

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
            customer_profit = reward1 * (config_4.MARGINS_1[pulled_arm] + reward2 * (config_4.MARGINS_2[promo]))
            daily_profits += customer_profit

        daily_rewards.append(daily_profits)

        ts_learner.compute_posterior(x_bar=current_daily_customers)

    ts_reward_per_experiment.append(daily_rewards)

ts_reward_per_experiment = np.array(ts_reward_per_experiment)

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(config_4.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.savefig(f"step4/plots/TS/TS_ExpRew_{config_4.N_EXPS}-{config_4.N_ARMS}.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(config_4.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.savefig(f"step4/plots/TS/TS_CumulativeExpRew_{config_4.N_EXPS}-{config_4.N_ARMS}.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.hlines(0, 0, 365, linestyles="dashed")
plt.plot(np.mean(config_4.OPT - ts_reward_per_experiment, axis=0), color='b')
plt.savefig(f"step4/plots/TS/TS_DailyRegret_{config_4.N_EXPS}-{config_4.N_ARMS}.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.hlines(0, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(config_4.OPT - ts_reward_per_experiment, axis=0)), color='c')
plt.savefig(f"step4/plots/UCB/UCB_CumulativeRegret_{config_4.N_EXPS}-{config_4.N_ARMS}.png", dpi=200)
plt.show()
