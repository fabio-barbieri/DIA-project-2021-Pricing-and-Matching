import numpy as np
import config

class Environment():
    def __init__(self, n_arms, first_conv_rates):
        self.n_arms = n_arms
        self.first_conv_rates = first_conv_rates

    def round(self, pulled_arm):
        rewards = [np.random.binomial(num, self.first_conv_rates[pulled_arm]) for num in config.num_customers]
        return rewards
