import config_5
from Environment_5 import *
import numpy as np
from tqdm import tqdm
from Learner_5 import *
import matplotlib.pyplot as plt

values_per_exp = []

for e in tqdm(range(config_5.N_EXPS)):
    env = Environment_5(cr1=config_5.CR1, 
                        cr2=config_5.CR2, 
                        num_customers=config_5.NUM_CUSTOMERS, 
                        sd_customers=config_5.SD_CUSTOMERS)
                        
    h_learner = Learner_5(tot_customers=np.sum(config_5.NUM_CUSTOMERS), 
                          promo_prob=config_5.PROMO_PROB, 
                          margin_1=config_5.MARGIN_1, 
                          margins_2=config_5.MARGINS_2, 
                          sd_customers=config_5.SD_CUSTOMERS)

    daily_values = []
    for t in range(config_5.T):
        # Build the matching at the start of the day with data from t-1
        matching_values, matching_mask = h_learner.compute_matching()
        daily_values.append(np.sum(matching_values))

        # Compute matching_prob
        matching_prob = h_learner.compute_matching_prob(matching_mask)

        # Observe the actual number of arrived customers and their order
        customer_arrivals, current_daily_customers = env.customers()

        # Simulate the rewards and update the betas
        for c_class in customer_arrivals:
            reward1, reward2, promo = env.round(c_class, matching_prob, h_learner.expected_customers)
            h_learner.update_betas(reward1, reward2, c_class, promo)

        # Compute posterior at the end of the day
        h_learner.compute_posterior(x_bar=current_daily_customers)

    values_per_exp.append(daily_values)

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected Reward")
plt.hlines(config_5.OPT, 0, 365, linestyles="dashed", color = 'darkgreen')
xx = range(365)
exp_rew_mean = np.mean(values_per_exp, axis=0)
exp_rew_sd = np.std(values_per_exp, axis=0)
upper = exp_rew_mean + 1.96 * exp_rew_sd / np.sqrt(config_5.N_EXPS)
lower = exp_rew_mean - 1.96 * exp_rew_sd / np.sqrt(config_5.N_EXPS)
plt.plot(exp_rew_mean, 'g')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'g')
plt.savefig(f"step5/plots/ExpRew_{config_5.N_EXPS}e-{config_5.N_ARMS}a.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Reward")
plt.hlines(config_5.OPT * 365, 0, 365, linestyles="dashed", color = 'darkred')
xx = range(365)
cum_rew_mean = np.cumsum(exp_rew_mean)
cum_rew_sd = np.cumsum(exp_rew_sd, axis=0)
upper = cum_rew_mean + 1.96 * cum_rew_sd / np.sqrt(config_5.N_EXPS)
lower = cum_rew_mean - 1.96 * cum_rew_sd / np.sqrt(config_5.N_EXPS)
plt.plot(cum_rew_mean, 'r')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'r')
plt.savefig(f"step5/plots/CumulativeExpRew_{config_5.N_EXPS}e-{config_5.N_ARMS}a.png", dpi=200)
plt.show()

#Nell'oraiginale non avevamo plottato il daily regret
plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily Regret")
plt.hlines(0, 0, 365, linestyles="dashed", color = 'k')
xx = range(365)
exp_reg_mean = np.mean(config_5.OPT - values_per_exp, axis=0)
exp_reg_sd = np.std(config_5.OPT - values_per_exp, axis=0)
upper = exp_reg_mean + 1.96 * exp_reg_sd / np.sqrt(config_5.N_EXPS)
lower = exp_reg_mean - 1.96 * exp_reg_sd / np.sqrt(config_5.N_EXPS)
plt.plot(exp_reg_mean, color='b')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'b')
plt.savefig(f"step5/plots/DailyRegret_{config_5.N_EXPS}e-{config_5.N_ARMS}a.png", dpi=200)
plt.show()

plt.figure(3, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.hlines(0, 0, 365, linestyles="dashed", color = 'k')
xx = range(365)
cum_reg_mean = np.cumsum(exp_reg_mean)
cum_reg_sd = np.cumsum(exp_reg_sd, axis=0)
upper = cum_reg_mean + 1.96 * cum_reg_sd / np.sqrt(config_5.N_EXPS)
lower = cum_reg_mean - 1.96 * cum_reg_sd / np.sqrt(config_5.N_EXPS)
plt.plot(cum_reg_mean, 'c')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'c')
plt.savefig(f"step5/plots/CumulativeRegret_{config_5.N_EXPS}e-{config_5.N_ARMS}a.png", dpi=200)
plt.show()


# # Plot the result
# plt.figure(0, figsize=(12, 7), dpi=200.0)
# plt.xlabel("t")
# plt.ylabel("Expected Profit")
# plt.hlines(config_5.OPT, 0, 365, linestyles="dashed")
# plt.plot(np.mean(values_per_exp, axis=0), 'g')
# plt.savefig(f"step5/plots/ExpProfit_{config_5.N_EXPS}e.png", dpi=200)
# plt.show()

# plt.figure(1, figsize=(12, 7), dpi=200.0)
# plt.xlabel("t")
# plt.ylabel("Cumulative Expected Profit")
# plt.hlines(config_5.OPT * 365, 0, 365, linestyles="dashed")
# plt.plot(np.cumsum(np.mean(values_per_exp, axis=0), axis=0), 'r')
# plt.savefig(f"step5/plots/CumulativeExpProfit_{config_5.N_EXPS}e.png", dpi=200)
# plt.show()

# plt.figure(2, figsize=(12, 7), dpi=200.0)
# plt.xlabel("t")
# plt.ylabel("Cumulative Regret")
# plt.hlines(0, 0, 365, linestyles="dashed")
# plt.plot(np.cumsum(np.mean(config_5.OPT - values_per_exp, axis=0)), color='c')
# plt.savefig(f"step5/plots/CumulativeRegret_{config_5.N_EXPS}e.png", dpi=200)
# plt.show()

