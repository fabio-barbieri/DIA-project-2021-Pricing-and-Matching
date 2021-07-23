import numpy as np

np.random.seed(1234)


class Environment_3:
    def __init__(self, n_arms, matching, cr1, cr2):
        self.n_arms = n_arms
        self.matching = matching
        self.cr1 = cr1
        self.cr2 = cr2

    def round(self, pulled_arm, c_class):
        # reward by a single customer
        reward1 = np.random.binomial(1, self.cr1[pulled_arm][c_class])
        promo = np.random.choice([0, 1, 2, 3], p=self.matching[c_class] / np.sum(self.matching[c_class]))
        reward2 = np.random.binomial(1, self.cr2[c_class][promo])

        return reward1, reward2, promo
