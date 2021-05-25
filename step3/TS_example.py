import config
import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *


n_arms = config.n_arms
cr1 = config.first_conversion_rates  # Bernoulli probability distributions
total_opt = np.dot(config.opt, config.num_customers)

n_exps = config.n_exps

T = config.T  # time horizon

ts_reward_per_experiment = []  # equal to collected reward

for e in range(n_exps):
    env = Environment(n_arms=n_arms, first_conv_rates=cr1)
    ts_learner = TS_Learner(n_arms=n_arms)

    for t in range(0, T):
        # Thompson Sampling learner
        pulled_arm = ts_learner.pull_arm()
        rewards = env.round(pulled_arm)
        ts_learner.update(pulled_arm, rewards)

    ts_reward_per_experiment.append(ts_learner.collected_rewards)

# Plot the results
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")

print(total_opt)
print()
print(ts_learner.collected_rewards)
print()
print(np.shape(ts_learner.collected_rewards))
print()
print(np.sum(ts_learner.collected_rewards, axis=1))

plt.plot(np.cumsum(np.mean(total_opt - np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'g')
#plt.plot(total_opt - np.sum(ts_learner.collected_rewards, axis=1), 'o', color='blue')
#plt.plot(np.cumsum(total_opt - np.sum(ts_learner.collected_rewards, axis=1)), 'r')
plt.legend(["TS"])
plt.show()

plt.plot(np.cumsum(np.mean(total_opt - np.sum(ts_reward_per_experiment, axis=2), axis=0)), 'r')