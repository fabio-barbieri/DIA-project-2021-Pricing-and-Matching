import numpy as np
import config
import utils


class Environment():
    def __init__(self, n_arms):
        self.n_arms = n_arms

    def round(self, pulled_arm):
        # For each class of customers, sample rewards from a binomial with mean the value of the i-th conv rate in the given pulled arm
        # i is an index related to the classes of customers
        pulled_margin = config.MARGINS_1[pulled_arm]
        rewards = [np.random.binomial(num, utils.cr1(pulled_margin, i)) for i, num in enumerate(config.NUM_CUSTOMERS)]
        return np.array(rewards)
