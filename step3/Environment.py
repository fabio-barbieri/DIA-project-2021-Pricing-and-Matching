import numpy as np
import config


class Environment():
    def __init__(self, n_arms, first_conv_rates):
        self.n_arms = n_arms
        self.conv_rates_1 = first_conv_rates

    def round(self, pulled_arm):
        # For each class of customers, sample rewards from a binomial with mean the value of the i-th conv rate in the given pulled arm
        # i is an index related to the classes of customers
        rewards = [np.random.binomial(num, self.conv_rates_1[pulled_arm][i]) for i, num in enumerate(config.NUM_CUSTOMERS)]
        return np.array(rewards)
