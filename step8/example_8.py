import config_8
import numpy as np
import matplotlib.pyplot as plt
from Non_Stationary_Environment_8 import *
from Learner_8 import *
from tqdm import tqdm

np.random.seed(1234)

ucb_reward_per_experiment = []  # Collected reward

for e in tqdm(range(config_8.N_EXPS)):
    env = Non_Stationary_Environment_8(num_customers=config_8.NUM_CUSTOMERS, 
                                      sd_customers=config_8.SD_CUSTOMERS, 
                                      n_arms_1=config_8.N_ARMS_1, 
                                      n_arms_2=config_8.N_ARMS_2, 
                                      cr1=config_8.CR1, 
                                      cr2=config_8.CR2,
                                      n_phases=len(config_8.SEASONS),
                                      horizon=config_8.T)

    learner = Learner_8(n_arms_1=config_8.N_ARMS_1, 
                        n_arms_2=config_8.N_ARMS_2, 
                        tot_customers=np.sum(config_8.NUM_CUSTOMERS),
                        promo_prob=config_8.PROMO_PROB,
                        sd_customers=config_8.SD_CUSTOMERS,
                        detection_params=config_8.DETECTION_PARAMS)

    daily_rewards = []
    for t in range(config_8.T):

        customer_arrivals, current_daily_customers = env.customers()

        daily_profits = 0
        for c_class in customer_arrivals:

            matching_prob, pulled_arm = learner.pull_arm()
            # print(matching_prob)
            reward1, reward2, promo = env.round(pulled_arm, c_class, matching_prob, learner.expected_customers)

            # reward1 * (margin1 + reward2 * margin2)
            curr_customer_profit = reward1 * (config_8.MARGINS_1[pulled_arm[0]] + reward2 * (config_8.MARGINS_2[pulled_arm[1]][promo]))
            normalizaiton_term = np.max(config_8.MARGINS_1) + np.max(config_8.MARGINS_2)
            normalized_curr_profit = curr_customer_profit / normalizaiton_term

            learner.update(pulled_arm, c_class, promo, normalized_curr_profit)

            daily_profits += curr_customer_profit

        daily_rewards.append(daily_profits)

        learner.compute_posterior(x_bar=current_daily_customers)

        env.update_day()

    ucb_reward_per_experiment.append(daily_rewards)


opt = np.dot(config_8.OPT, [91, 91, 91, 92])

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected Reward")
plt.step([0, 91, 182, 273, 365], np.insert(config_8.OPT, -1, config_8.OPT[-1]), where='post', linestyle='dashed', color='black')
plt.axis([0, 365, 0, 18000])
xx = range(365)
exp_rew_mean = np.mean(ucb_reward_per_experiment, axis=0)
exp_rew_sd = np.std(ucb_reward_per_experiment, axis=0)
upper = exp_rew_mean + 1.96 * exp_rew_sd / np.sqrt(config_8.N_EXPS)
lower = exp_rew_mean - 1.96 * exp_rew_sd / np.sqrt(config_8.N_EXPS)
plt.plot(exp_rew_mean, 'g')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'g')
plt.savefig(f"step8/plots/setting{config_8.SETTING}/ExpRew_set{config_8.SETTING}-{config_8.N_EXPS}e-{config_8.N_ARMS_1}a1-{config_8.N_ARMS_2}a2.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Reward")
plt.hlines(opt, 0, 365, linestyles="dashed", colors='black')
xx = range(365)
cum_rew_mean = np.cumsum(exp_rew_mean)
cum_rew_sd = np.cumsum(exp_rew_sd, axis=0)
upper = cum_rew_mean + 1.96 * cum_rew_sd / np.sqrt(config_8.N_EXPS)
lower = cum_rew_mean - 1.96 * cum_rew_sd / np.sqrt(config_8.N_EXPS)
plt.plot(cum_rew_mean, 'r')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'r')
plt.savefig(f"step8/plots/setting{config_8.SETTING}/CumulativeExpRew_set{config_8.SETTING}-{config_8.N_EXPS}e-{config_8.N_ARMS_1}a1-{config_8.N_ARMS_2}a2.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily Regret")
plt.hlines(0, 0, 365, linestyles="dashed", colors='black')
xx = range(365)
opt = np.repeat(config_8.OPT, [91, 91, 91, 92])
plottable = (-1) * (np.array(ucb_reward_per_experiment, dtype=int) - opt)
exp_reg_mean = np.mean(plottable, axis=0)
exp_reg_sd = np.std(plottable, axis=0)
upper = exp_reg_mean + 1.96 * exp_reg_sd / np.sqrt(config_8.N_EXPS)
lower = exp_reg_mean - 1.96 * exp_reg_sd / np.sqrt(config_8.N_EXPS)
plt.plot(exp_reg_mean, color='b')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'b')
plt.savefig(f"step8/plots/setting{config_8.SETTING}/DailyRegret_set{config_8.SETTING}-{config_8.N_EXPS}e-{config_8.N_ARMS_1}a1-{config_8.N_ARMS_2}a2.png", dpi=200)
plt.show()

plt.figure(3, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
xx = range(365)
cum_reg_mean = np.cumsum(exp_reg_mean)
cum_reg_sd = np.cumsum(exp_reg_sd, axis=0)
upper = cum_reg_mean + 1.96 * cum_reg_sd / np.sqrt(config_8.N_EXPS)
lower = cum_reg_mean - 1.96 * cum_reg_sd / np.sqrt(config_8.N_EXPS)
plt.plot(cum_reg_mean, 'c')
plt.fill_between(xx, upper, lower, alpha = 0.2, color = 'c')
plt.savefig(f"step8/plots/setting{config_8.SETTING}/CumulativeRegret_set{config_8.SETTING}-{config_8.N_EXPS}e-{config_8.N_ARMS_1}a1-{config_8.N_ARMS_2}a2.png", dpi=200)
plt.show()