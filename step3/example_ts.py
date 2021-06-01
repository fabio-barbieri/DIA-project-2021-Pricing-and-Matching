import config
import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *
from tqdm import tqdm


TOTAL_OPT = np.dot(config.OPT, config.NUM_CUSTOMERS)

ts_reward_per_experiment = []  # Collected reward
for e in tqdm(range(config.N_EXPS)):
    env = Environment(n_arms=config.N_ARMS, cr1=config.CR1)
    ts_learner = TS_Learner(n_arms=config.N_ARMS)

    for t in range(config.T):
        # Thompson Sampling
        pulled_arm = ts_learner.pull_arm()
        rewards = env.round(pulled_arm)
        ts_learner.update(pulled_arm, rewards)

    ts_reward_per_experiment.append(ts_learner.collected_rewards)

print(np.shape(ts_learner.collected_rewards))

# Plot the results
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(TOTAL_OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(np.sum(ts_reward_per_experiment, axis=2), axis=0), 'g')
plt.show()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.plot(np.cumsum(np.mean(np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'r')
plt.show()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.plot(np.mean(TOTAL_OPT - np.sum(ts_reward_per_experiment, axis=2), axis=0), color='b')
plt.show()

#plt.plot(np.cumsum(np.mean(TOTAL_OPT - np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'g')  # Cumulative mean of regrets
#plt.plot(TOTAL_OPT - np.sum(ts_learner.collected_rewards, axis=1), color='blue')  # Daily regret
#plt.plot(np.cumsum(TOTAL_OPT - np.sum(ts_learner.collected_rewards, axis=1)), 'r')  # Cumulative daily regret

#plt.plot(np.cumsum(np.mean(TOTAL_OPT - np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'r')