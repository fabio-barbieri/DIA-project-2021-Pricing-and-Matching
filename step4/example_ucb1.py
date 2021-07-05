import config
import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from UCB1_Learner import *
from tqdm import tqdm

np.random.seed(1234)

ucb1_reward_per_experiment = []  # Collected reward

for e in tqdm(range(config.N_EXPS)):
    env = Environment(n_arms=config.N_ARMS, cr1=config.CR1, cr2=config.CR2)
    ucb1_learner = UCB1_Learner(n_arms=config.N_ARMS)

    daily_rewards = []

    for t in range(config.T):

        customer_arrivals, current_daily_customers = env.customers()

        daily_profits = 0
        for c_class in customer_arrivals:
            # UCB1
            pulled_arm = ucb1_learner.pull_arm()
            reward1, reward2, promo = env.round(pulled_arm, c_class, current_daily_customers)  # questo deve diventare 0 o 1
            ucb1_learner.update(pulled_arm, reward1, reward2, c_class, promo)  # update solo della beta della classe del cliente corrente

            # reward * (margin1 + promo * margin2 * conv2[pulled_arm])
            avg_customer_profit = reward1 * (config.MARGINS_1[pulled_arm] + reward2 * config.MARGINS_2[promo])
            daily_profits += avg_customer_profit

        daily_rewards.append(daily_profits)

        #ucb1_learner.update_expected_customers(current_daily_customers, t + 1)
        ucb1_learner.compute_posterior(x_bar=current_daily_customers)

    ucb1_reward_per_experiment.append(daily_rewards)

print(np.shape(daily_rewards))
print(np.shape(ucb1_reward_per_experiment))

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(config.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ucb1_reward_per_experiment, axis=0), 'g')
plt.savefig("expected_reward.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(config.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ucb1_reward_per_experiment, axis=0)), 'r')
plt.savefig("cumulative_expected_reward.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.plot(np.mean(config.OPT - ucb1_reward_per_experiment, axis=0), color='b')
plt.savefig("daily_regret.png", dpi=200)
plt.show()
