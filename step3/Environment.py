import numpy as np
import config
import utils


class Environment():
    def __init__(self, n_arms, cr1):
        self.n_arms = n_arms
        self.cr1 = cr1

    def round(self, pulled_arm):
        # For each class of customers, sample rewards from a binomial with mean the value of the i-th conv rate in the given pulled arm
        # i is an index related to the classes of customers
        rewards = [np.random.binomial(num, self.cr1[pulled_arm][i]) for i, num in enumerate(config.NUM_CUSTOMERS)]
        return np.array(rewards)
