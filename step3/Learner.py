# THIS IS GOING TO BE THE SUPERCLASS OF THE THOMPSON SAMPLING AND GREEDY ALGORITHMS LEARNERS
import numpy as np

class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(n_arms)]
        # self.reward_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = []

    def update_observations(self, pulled_arm, rewards):
        # reward_per_arm is an array of 4-dimensional arrays
        # collected rewards has 4 dimensional array per day
        self.rewards_per_arm[pulled_arm].append(rewards)
        self.collected_rewards.append(rewards)
