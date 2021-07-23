import config_6
import numpy as np
import matplotlib.pyplot as plt
from Environment_6 import *
from Learner_6 import *
from tqdm import tqdm

np.random.seed(1234)

ts_reward_per_experiment = []  # Collected reward

opt = []

for e in tqdm(range(config_6.N_EXPS)):
    env = Environment_6(num_customers=config_6.NUM_CUSTOMERS, 
                        sd_customers=config_6.SD_CUSTOMERS, 
                        n_arms_1=config_6.N_ARMS_1, 
                        n_arms_2=config_6.N_ARMS_2, 
                        cr1=config_6.CR1, cr2=config_6.CR2)
    learner = Learner_6(n_arms_1=config_6.N_ARMS_1, n_arms_2=config_6.N_ARMS_2)

    daily_rewards = []

    for t in range(config_6.T):

        customer_arrivals, current_daily_customers = env.customers()

        daily_profits = 0
        for c_class in customer_arrivals:

            matching_prob, pulled_arm = learner.pull_arm()
            # print(matching_prob)
            reward1, reward2, promo = env.round(pulled_arm, c_class, matching_prob, learner.expected_customers)
            learner.update(pulled_arm, reward1, reward2, c_class, promo)

            # reward1 * (margin1 + reward2 * margin2)
            curr_customer_profit = reward1 * (config_6.MARGINS_1[pulled_arm[0]] + reward2 * (config_6.MARGINS_2[pulled_arm[1]][promo]))
            daily_profits += curr_customer_profit

        daily_rewards.append(daily_profits)

        learner.compute_posterior(x_bar=current_daily_customers)

    ts_reward_per_experiment.append(daily_rewards)

    # constructing matrix in order to
    # opt_price_1, opt_price_2 = np.unravel_index(np.argmax(learner.n_pulled_couple), learner.n_pulled_couple.shape)
    # opt_value = learner.n_pulled_couple[opt_price_1, opt_price_2]

    # opt.append((opt_price_1, opt_price_2, opt_value))


# def build_matrix_opt(idx1, idx2):
#     matrix_dim = config_6.TOT_CUSTOMERS

#     matrix = np.zeros((4, 0))
#     profit = np.multiply(config_6.CR1[idx1].reshape((4, 1)), config_6.MARGINS_1[idx1] + np.multiply(config_6.CR2[idx2],
#                          config_6.MARGINS_2[idx2]))

#     n_promos = (config_6.PROMO_PROB[1:] * matrix_dim).astype(int)
#     n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))

#     # repeat columns
#     matrix = np.column_stack([matrix, np.repeat(profit, n_promos, axis=1)])

#     # repeat rows
#     matrix = np.repeat(matrix, config_6.NUM_CUSTOMERS, axis=0)

#     return matrix


# # compute optimum daily reward
# opt_tuple = max(opt, key=lambda x: x[2])
# opt_mask, OPT = hungarian_algorithm(build_matrix_opt(opt_tuple[0], opt_tuple[1]))

# Plot the results
plt.figure(0, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Expected reward")
plt.hlines(config_6.OPT, 0, 365, linestyles="dashed")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'g')
plt.savefig("expected_reward.png", dpi=200)
plt.show()

plt.figure(1, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Cumulative expected reward")
plt.hlines(config_6.OPT * 365, 0, 365, linestyles="dashed")
plt.plot(np.cumsum(np.mean(ts_reward_per_experiment, axis=0)), 'r')
plt.savefig("cumulative_expected_reward.png", dpi=200)
plt.show()

plt.figure(2, figsize=(12, 7), dpi=200.0)
plt.xlabel("t")
plt.ylabel("Daily regret")
plt.plot(np.mean(config_6.OPT - ts_reward_per_experiment, axis=0), color='b')
plt.savefig("daily_regret.png", dpi=200)
plt.show()
