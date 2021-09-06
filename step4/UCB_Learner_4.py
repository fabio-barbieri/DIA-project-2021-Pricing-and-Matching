from Learner_4 import *
import numpy as np

np.random.seed(1234)

class UCB_Learner_4(Learner_4):

    def __init__(self, n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2):
        super().__init__(n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.t = 0
        self.n_pulled_arm = np.zeros(n_arms, dtype=int)

    def pull_arm(self):
        if self.t < self.n_arms:
            idx = self.t
        else:
            upper_bound = self.empirical_means + self.confidence
            idx = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        
        return idx

    def update(self, pulled_arm, reward1, reward2, promo):
        self.t += 1  # Increment the counter of entered customers

        profit = reward1 * (self.margins_1[pulled_arm] + reward2 * self.margins_2[promo])

        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + profit) / self.t
        self.confidence = ((2 * np.log(self.t) / np.maximum(1, self.n_pulled_arm)) ** 0.5) * profit

        self.n_pulled_arm[pulled_arm] += 1


