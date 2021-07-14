import numpy as np
from Learner_3 import *

np.random.seed(1234)

class UCB1_Learner_3(Learner_3):

    def __init__(self, n_arms, num_customers, margins_1, matching, margins_2, cr2):
        super().__init__(n_arms, num_customers, margins_1, matching, margins_2, cr2)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.t = 0
        self.n_pulled_arm = np.zeros(n_arms, dtype=int)

    #def profit(self, i, upper_bound):
    #    cr1 = upper_bound[i]                                                       # 4x1
    #    margins1 = self.margins_1[i]                                               # 1x1
    #    cr2 = self.cr2                                                             # 4x4
    #    matching_prob = self.matching / np.expand_dims(self.num_customers, axis=1) # 4x4
    #    margins2 = self.margins_2                                                  # 4x1

    #    a = cr1 * (margins1 + np.dot(cr2 * matching_prob, margins2)) #   4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) =
                                                                     # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                     # = 4x1 * (1x1 + 4x1) = 
                                                                     # = 4x1 * 4x1 = 
                                                                     # = 4x1
    #   return np.dot(a, self.num_customers)

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

        # scaling confidence multiplying it for the profit, in order to assure consistent exploration of arms
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + profit) / self.t
        self.confidence = ((2 * np.log(self.t) / np.maximum(1, self.n_pulled_arm)) ** 0.5) * profit

        self.n_pulled_arm[pulled_arm] += 1
