import config_7
import numpy as np
import matplotlib.pyplot as plt
from Environment_7 import *
from Learner_7 import *
from tqdm import tqdm

np.random.seed(1234)

ts_reward_per_experiment = []  # Collected reward

opt = []

for e in tqdm(range(config_7.N_EXPS)):
    env = Environment_7(num_customers=config_7.NUM_CUSTOMERS, 
                        sd_customers=config_7.SD_CUSTOMERS, 
                        n_arms_1=config_7.N_ARMS_1, 
                        n_arms_2=config_7.N_ARMS_2, 
                        cr1=config_7.CR1, 
                        cr2=config_7.CR2,
                        n_phases=config_7.N_PHASES,
                        horizon=config_7.T)
    learner = Learner_7(n_arms_1=config_7.N_ARMS_1, n_arms_2=config_7.N_ARMS_2, window_size=config_7.WINDOW_SIZE)

    daily_rewards = []

    for t in range(config_7.T):

        customer_arrivals, current_daily_customers = env.customers()

        daily_profits = 0
        for c_class in customer_arrivals:

            matching_prob, pulled_arm = learner.pull_arm()
            # print(matching_prob)
            reward1, reward2, promo = env.round(pulled_arm, c_class, matching_prob, learner.expected_customers)
            learner.update(pulled_arm, reward1, reward2, c_class, promo)

            # reward1 * (margin1 + reward2 * margin2)
            curr_customer_profit = reward1 * (config_7.MARGINS_1[pulled_arm[0]] + reward2 * (config_7.MARGINS_2[pulled_arm[1]][promo]))
            daily_profits += curr_customer_profit

        daily_rewards.append(daily_profits)

        learner.compute_posterior(x_bar=current_daily_customers)

        env.update_day()

    ts_reward_per_experiment.append(daily_rewards)

opt = np.dot(config_7.OPT, [91, 91, 91, 92])
# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.step([0, 91, 182, 273, 365], np.insert(config_7.OPT, -1, config_7.OPT[-1]), where='post', linestyle='dashed')
plt.axis([0, 365, 0, 18000])
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.savefig("step7/plots/expected_reward.png", dpi=200)
plt.show()


plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(opt, 0, 365, linestyle="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.savefig("step7/plots/cumulative_expected_reward.png", dpi=200)
plt.show()


plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily regret")
opt = np.repeat(config_7.OPT, [91, 91, 91, 92])
plottable = (-1) * (np.array(ts_reward_per_experiment, dtype=int) - opt)
plt.plot(np.mean(plottable, axis=0), color='b')
plt.hlines(0, 0, 365, linestyle='dashed')
plt.savefig("step7/plots/daily_regret.png", dpi=200)
plt.show()
