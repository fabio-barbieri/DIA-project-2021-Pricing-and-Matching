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
                        params=config_8.DETECTION_PARAMS)

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
plt.step([0, 91, 182, 273, 365], np.insert(config_8.OPT, -1, config_8.OPT[-1]), where='post', linestyle="dashed")
plt.axis([0, 365, 0, 18000])
plt.plot(np.mean(ucb_reward_per_experiment, axis=0), 'g')
plt.savefig(f"step8/plots/ExpRew_{config_8.N_EXPS}e-{config_8.N_ARMS_1}a1-{config_8.N_ARMS_2}a2.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Expected Reward")
plt.hlines(opt, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ucb_reward_per_experiment, axis=0)), 'r')
plt.savefig(f"step8/plots/CumulativeExpRew_{config_8.N_EXPS}e-{config_8.N_ARMS_1}a1-{config_8.N_ARMS_2}a2.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily Regret")
opt = np.repeat(config_8.OPT, [91, 91, 91, 92])
plottable = (-1) * (np.array(ucb_reward_per_experiment, dtype=int) - opt)
plt.plot(np.mean(plottable, axis=0), color='b')
plt.hlines(0, 0, 365, linestyles="dashed")
plt.savefig(f"step8/plots/DailyRegret_{config_8.N_EXPS}e-{config_8.N_ARMS_1}a1-{config_8.N_ARMS_2}a2.png", dpi=200)
plt.show()

plt.figure(3, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.hlines(0, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(plottable, axis=0)), color='c')
plt.savefig(f"step8/plots/CumulativeRegret_{config_8.N_EXPS}e-{config_8.N_ARMS_1}a1-{config_8.N_ARMS_2}a2.png", dpi=200)
plt.show()
