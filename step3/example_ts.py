import config
import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *


N_ARMS = config.N_ARMS
CR1 = config.CONV_RATES_1
TOTAL_OPT = np.dot(config.OPT, config.NUM_CUSTOMERS)

N_EXPS = config.N_EXPS
T = config.T

ts_reward_per_experiment = []  # Collected reward
for e in range(N_EXPS):
    env = Environment(n_arms=N_ARMS, first_conv_rates=CR1)
    ts_learner = TS_Learner(n_arms=N_ARMS)

    for t in range(T):
        # Thompson Sampling
        pulled_arm = ts_learner.pull_arm()
        rewards = env.round(pulled_arm)
        ts_learner.update(pulled_arm, rewards)

    ts_reward_per_experiment.append(ts_learner.collected_rewards)

# Plot the results
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(TOTAL_OPT - np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'g')
plt.plot(TOTAL_OPT - np.sum(ts_learner.collected_rewards, axis=1), 'o', color='blue')
plt.plot(np.cumsum(TOTAL_OPT - np.sum(ts_learner.collected_rewards, axis=1)), 'r')
plt.legend(["TS"])
plt.show()

plt.plot(np.cumsum(np.mean(TOTAL_OPT - np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'r')