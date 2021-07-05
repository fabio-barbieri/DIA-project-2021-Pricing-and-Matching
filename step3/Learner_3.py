# THIS IS GOING TO BE THE SUPERCLASS OF THE THOMPSON SAMPLING AND GREEDY ALGORITHMS LEARNERS
import numpy as np

class Learner_3:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = []

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)  # ... (for UCB ???)
        self.collected_rewards.append(reward)  # List of N_EXPS arrays of shape (T, 4)
