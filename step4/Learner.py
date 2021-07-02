# THIS IS GOING TO BE THE SUPERCLASS OF THE THOMPSON SAMPLING AND GREEDY ALGORITHMS LEARNERS
import numpy as np
import config

class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = []
        self.m = np.array([config.TOT_CUSTOMERS // 4 for _ in range(4)]) # Non-informative prior
        self.s_2 = np.ones((4, 1))

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)  # ... (for UCB ???)
        self.collected_rewards.append(reward)  # List of N_EXPS arrays of shape (T, 4)

    def compute_posterior(x_bar):
        sigma_2 = config.SD_CUSTOMERS ** 2
        m = self.m
        s_2 = self.s_2

        self.m = (s_2 * x_bar + m * sigma_2) / (s_2 + sigma_2)
        self.s_2 = (s_2 * sigma_2) / (s_2 + sigma_2)