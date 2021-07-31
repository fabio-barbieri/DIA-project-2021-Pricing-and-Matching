import config_3
import numpy as np
import matplotlib.pyplot as plt
from Environment_3 import *
from TS_Learner_3 import *
from tqdm import tqdm

np.random.seed(1234)


ts_reward_per_experiment = []  # Collected reward

tot_customers = np.sum(config_3.NUM_CUSTOMERS)
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
                        
    ts_learner = TS_Learner_3(n_arms=config_3.N_ARMS, 
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
            # Thompson Sampling
            pulled_arm = ts_learner.pull_arm()
            reward1, reward2, promo = env.round(pulled_arm, c_class)  # questo deve diventare 0 o 1
            ts_learner.update(pulled_arm, reward1, c_class)  # update solo della beta della classe del cliente corrente

            # reward * (margin1 + promo * margin2 * conv2[pulled_arm])
            customer_profit = reward1 * (config_3.MARGINS_1[pulled_arm] + reward2 * config_3.MARGINS_2[promo])
            daily_profits += customer_profit

        daily_rewards.append(daily_profits)

    ts_reward_per_experiment.append(daily_rewards)

ts_reward_per_experiment = np.array(ts_reward_per_experiment)


# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected Reward")
plt.hlines(config_3.OPT, 0, 365, linestyles="dashed", color = 'darkgreen')
xx = range(365)
exp_rew_mean = np.mean(ts_reward_per_experiment, axis=0)
exp_rew_sd = np.std(ts_reward_per_experiment, axis=0)
upper = exp_rew_mean + 1.96 * exp_rew_sd / np.sqrt(config_3.N_EXPS)
lower = exp_rew_mean - 1.96 * exp_rew_sd / np.sqrt(config_3.N_EXPS)
plt.plot(exp_rew_mean, 'g')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'g')
plt.savefig(f"step3/plots/TS/TS_ExpRew_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Reward")
plt.hlines(config_3.OPT * 365, 0, 365, linestyles="dashed", color = 'darkred')
xx = range(365)
cum_rew_mean = np.cumsum(exp_rew_mean)
cum_rew_sd = np.cumsum(exp_rew_sd, axis=0)
upper = cum_rew_mean + 1.96 * cum_rew_sd / np.sqrt(config_3.N_EXPS)
lower = cum_rew_mean - 1.96 * cum_rew_sd / np.sqrt(config_3.N_EXPS)
plt.plot(cum_rew_mean, 'r')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'r')
plt.savefig(f"step3/plots/TS/TS_CumulativeExpRew_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily Regret")
plt.hlines(0, 0, 365, linestyles="dashed", color = 'k')
xx = range(365)
exp_reg_mean = np.mean(config_3.OPT - ts_reward_per_experiment, axis=0)
exp_reg_sd = np.std(config_3.OPT - ts_reward_per_experiment, axis=0)
upper = exp_reg_mean + 1.96 * exp_reg_sd / np.sqrt(config_3.N_EXPS)
lower = exp_reg_mean - 1.96 * exp_reg_sd / np.sqrt(config_3.N_EXPS)
plt.plot(exp_reg_mean, color='b')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'b')
plt.savefig(f"step3/plots/TS/TS_DailyRegret_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
plt.show()

plt.figure(3, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.hlines(0, 0, 365, linestyles="dashed", color = 'k')
xx = range(365)
cum_reg_mean = np.cumsum(exp_reg_mean)
cum_reg_sd = np.cumsum(exp_reg_sd, axis=0)
upper = cum_reg_mean + 1.96 * cum_reg_sd / np.sqrt(config_3.N_EXPS)
lower = cum_reg_mean - 1.96 * cum_reg_sd / np.sqrt(config_3.N_EXPS)
plt.plot(cum_reg_mean, 'c')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'c')
plt.savefig(f"step3/plots/TS/TS_CumulativeRegret_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
plt.show()



# # Plot the results
# plt.figure(0, figsize=(12, 7), dpi=200.0)
# plt.xlabel("t")
# plt.ylabel("Expected Reward")
# plt.hlines(config_3.OPT, 0, 365, linestyles="dashed")
# plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
# plt.savefig(f"step3/plots/TS/TS_ExpRew_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
# plt.show()

# plt.figure(1, figsize=(12, 7), dpi=200.0)
# plt.xlabel("t")
# plt.ylabel("Cumulative Expected Reward")
# plt.hlines(config_3.OPT * 365, 0, 365, linestyles="dashed")
# plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
# plt.savefig(f"step3/plots/TS/TS_CumulativeExpRew_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
# plt.show()

# plt.figure(2, figsize=(12, 7), dpi=200.0)
# plt.xlabel("t")
# plt.ylabel("Daily Regret")
# plt.hlines(0, 0, 365, linestyles="dashed")
# plt.plot(np.mean(config_3.OPT - ts_reward_per_experiment, axis=0), color='b')
# plt.savefig(f"step3/plots/TS/TS_DailyRegret_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
# plt.show()

# plt.figure(2, figsize=(12, 7), dpi=200.0)
# plt.xlabel("t")
# plt.ylabel("Cumulative Regret")
# plt.hlines(0, 0, 365, linestyles="dashed")
# plt.plot(np.cumsum(np.mean(config_3.OPT - ts_reward_per_experiment, axis=0)), color='c')
# plt.savefig(f"step3/plots/TS/TS_CumulativeRegret_{config_3.N_EXPS}e-{config_3.N_ARMS}a.png", dpi=200)
# plt.show()