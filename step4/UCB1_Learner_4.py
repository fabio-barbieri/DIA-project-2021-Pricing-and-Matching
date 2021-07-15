from Learner_4 import *
import numpy as np


class UCB1_Learner_4(Learner_4):

    def __init__(self, n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2):
        super().__init__(n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.beta_cr2 = np.ones((4, 4, 2))
        self.t = 0
        self.n_pulled_arm = np.zeros(n_arms, dtype=int)

    def pull_arm(self):
        if self.t < self.n_arms:
            idx = self.t
        else:
            upper_bound = self.empirical_means + self.confidence
            idx = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        
        return idx

    def update(self, pulled_arm, reward1, reward2, c_class, promo):
        self.t += 1  # Increment the counter of entered customers

        profit = reward1 * (self.margins_1[pulled_arm] + reward2 * self.margins_2[promo])

        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + profit) / self.t
        self.confidence = ((2 * np.log(self.t) / np.maximum(1, self.n_pulled_arm)) ** 0.5) * profit

        # update beta parameters associated with conversion rates on product 2, if the first item has been bought
        if reward1 == 1:
            self.beta_cr2[c_class, promo, 0] = self.beta_cr2[c_class, promo, 0] + reward2
            self.beta_cr2[c_class, promo, 1] = self.beta_cr2[c_class, promo, 1] + (1.0 - reward2)

        self.n_pulled_arm[pulled_arm] += 1


