import config_5
from Environment_5 import *
import numpy as np
from tqdm import tqdm
from Learner_5 import *
import matplotlib.pyplot as plt

np.random.seed(1234)

values_per_exp = []
rewards_per_exp = []
for e in tqdm(range(config_5.N_EXPS)):
    env = Environment_5(cr1=config_5.CR1, 
                        cr2=config_5.CR2, 
                        num_customers=config_5.NUM_CUSTOMERS, 
                        sd_customers=config_5.SD_CUSTOMERS)
                        
    learner = Learner_5(tot_customers=np.sum(config_5.NUM_CUSTOMERS), 
                          promo_prob=config_5.PROMO_PROB, 
                          margin_1=config_5.MARGIN_1, 
                          margins_2=config_5.MARGINS_2, 
                          sd_customers=config_5.SD_CUSTOMERS)

    daily_values = [] # Daily values obtained from the matching
    daily_profits = [] # Actual daily values of the profits
    for t in range(config_5.T):
        # Build the matching at the start of the day with data from t-1
        matching_values, matching_mask = learner.compute_matching()
        daily_values.append(np.sum(matching_values))

        # Compute matching_prob
        matching_prob = learner.compute_matching_prob(matching_mask)

        # Observe the actual number of arrived customers and their order
        customer_arrivals, current_daily_customers = env.customers()

        # Initialise the daily profit
        daily_profit = 0

        # Simulate the rewards and update the betas
        for c_class in customer_arrivals:
            reward1, reward2, promo = env.round(c_class, matching_prob, learner.expected_customers)
            learner.update_betas(reward1, reward2, c_class, promo)
            # Compute the customer profit
            daily_profit += reward1 * (config_5.MARGIN_1 + reward2 * config_5.MARGINS_2[promo])

        # Compute posterior at the end of the day
        learner.compute_posterior(x_bar=current_daily_customers)
        
        # Append the daily profit
        daily_profits.append(daily_profit)

    values_per_exp.append(daily_values)
    rewards_per_exp.append(daily_profits)

values_per_exp = np.array(values_per_exp)
rewards_per_exp = np.array(rewards_per_exp)

# Plot the results for the reward
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected Reward")
plt.hlines(config_5.OPT, 0, 365, linestyles="dashed", colors='black')
xx = range(365)
exp_rew_mean = np.mean(rewards_per_exp, axis=0)
exp_rew_sd = np.std(rewards_per_exp, axis=0)
upper = exp_rew_mean + 1.96 * exp_rew_sd / np.sqrt(config_5.N_EXPS)
lower = exp_rew_mean - 1.96 * exp_rew_sd / np.sqrt(config_5.N_EXPS)
plt.plot(exp_rew_mean, 'g')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'g')
plt.savefig(f"step5/plots/setting{config_5.SETTING}/ExpRew_set{config_5.SETTING}-{config_5.N_EXPS}e.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Reward")
plt.hlines(config_5.OPT * 365, 0, 365, linestyles="dashed", colors='black')
xx = range(365)
cum_rew_mean = np.cumsum(exp_rew_mean)
cum_rew_sd = np.cumsum(exp_rew_sd, axis=0)
upper = cum_rew_mean + 1.96 * cum_rew_sd / np.sqrt(config_5.N_EXPS)
lower = cum_rew_mean - 1.96 * cum_rew_sd / np.sqrt(config_5.N_EXPS)
plt.plot(cum_rew_mean, 'r')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'r')
plt.savefig(f"step5/plots/setting{config_5.SETTING}/CumulativeExpRew_set{config_5.SETTING}-{config_5.N_EXPS}e.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily Regret")
plt.hlines(0, 0, 365, linestyles="dashed", colors='black')
xx = range(365)
exp_reg_mean = np.mean(config_5.OPT - rewards_per_exp, axis=0)
exp_reg_sd = np.std(config_5.OPT - rewards_per_exp, axis=0)
upper = exp_reg_mean + 1.96 * exp_reg_sd / np.sqrt(config_5.N_EXPS)
lower = exp_reg_mean - 1.96 * exp_reg_sd / np.sqrt(config_5.N_EXPS)
plt.plot(exp_reg_mean, color='b')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'b')
plt.savefig(f"step5/plots/setting{config_5.SETTING}/DailyRegret_set{config_5.SETTING}-{config_5.N_EXPS}e.png", dpi=200)
plt.show()

plt.figure(3, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
xx = range(365)
cum_reg_mean = np.cumsum(exp_reg_mean)
cum_reg_sd = np.cumsum(exp_reg_sd, axis=0)
upper = cum_reg_mean + 1.96 * cum_reg_sd / np.sqrt(config_5.N_EXPS)
lower = cum_reg_mean - 1.96 * cum_reg_sd / np.sqrt(config_5.N_EXPS)
plt.plot(cum_reg_mean, 'c')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'c')
plt.savefig(f"step5/plots/setting{config_5.SETTING}/CumulativeRegret_set{config_5.SETTING}-{config_5.N_EXPS}e.png", dpi=200)
plt.show()


# Plot the results for the matching values
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected Matching Value")
plt.hlines(config_5.OPT, 0, 365, linestyles="dashed", colors='black')
xx = range(365)
exp_val_mean = np.mean(values_per_exp, axis=0)
exp_val_sd = np.std(values_per_exp, axis=0)
upper = exp_val_mean + 1.96 * exp_val_sd / np.sqrt(config_5.N_EXPS)
lower = exp_val_mean - 1.96 * exp_val_sd / np.sqrt(config_5.N_EXPS)
plt.plot(exp_val_mean, 'deeppink')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'deeppink')
plt.savefig(f"step5/plots/setting{config_5.SETTING}/ExpMatchVal_set{config_5.SETTING}-{config_5.N_EXPS}e.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Matching Value")
plt.hlines(config_5.OPT * 365, 0, 365, linestyles="dashed", colors='black')
xx = range(365)
cum_val_mean = np.cumsum(exp_val_mean)
cum_val_sd = np.cumsum(exp_val_sd, axis=0)
upper = cum_val_mean + 1.96 * cum_val_sd / np.sqrt(config_5.N_EXPS)
lower = cum_val_mean - 1.96 * cum_val_sd / np.sqrt(config_5.N_EXPS)
plt.plot(cum_val_mean, 'darkorange')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'darkorange')
plt.savefig(f"step5/plots/setting{config_5.SETTING}/CumulativeExpMatchVal_set{config_5.SETTING}-{config_5.N_EXPS}e.png", dpi=200)
plt.show()