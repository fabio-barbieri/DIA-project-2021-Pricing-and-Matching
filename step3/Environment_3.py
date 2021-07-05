import numpy as np
import config_3
import utils_3


class Environment_3():
    def __init__(self, n_arms, cr1):
        self.n_arms = n_arms
        self.cr1 = cr1

    def round(self, pulled_arm, c_class):
        # reward by a single customer
        reward = np.random.binomial(1, self.cr1[pulled_arm][c_class])
        return reward
