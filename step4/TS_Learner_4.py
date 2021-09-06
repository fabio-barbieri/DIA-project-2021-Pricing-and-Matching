import numpy as np
from Learner_4 import *

np.random.seed(1234)

class TS_Learner_4(Learner_4):

    def __init__(self, n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2):
        super().__init__(n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2)
        # Array used to store the values of parameters alpha and beta for the
        # beta distributions related at each class of customers, for each possible arm
        self.beta_parameters = np.ones((n_arms, 4, 2))  # n_arms = num of item 1 prices
                                                        # 4 = num of customer classes
                                                        # 2 = amount of parameters for the beta distribution (alpa, beta)
        self.beta_cr2 = np.ones((4, 4, 2))

    def profit(self, i, arm):
        cr1 = np.random.beta(arm[:, 0], arm[:, 1])                                                                             # 4x1
        margin1 = self.margins_1[i]                                                                                            # 1x1
        cr2 = np.random.beta(self.beta_cr2[:, :, 0], self.beta_cr2[:, :, 1])                                                   # 4x4
        matching_prob = self.matching_prob / np.expand_dims(self.expected_customers, axis=1) * np.sum(self.expected_customers) # 4x4
        margins2 = self.margins_2                                                                                              # 4x1

        a = cr1 * (margin1 + np.dot(cr2 * matching_prob, margins2)) #   4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                    # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                    # = 4x1 * (1x1 + 4x1) = 
                                                                    # = 4x1 * 4x1 = 
                                                                    # = 4x1
        return np.dot(a, self.expected_customers)

    def pull_arm(self):
        # Pull the arm that maximizes the profit w.r.t. all the classes of customers and the beta distributions
        profits = np.array([self.profit(i, arm) for i, arm in enumerate(self.beta_parameters)])
        return np.argmax(profits)

    def update(self, pulled_arm, reward1, reward2, c_class, promo):
        # Update the parameters of the betas according to the rewards and considering that the average num
        # of customers per class must be considered
        self.beta_parameters[pulled_arm, c_class, 0] = self.beta_parameters[pulled_arm, c_class, 0] + reward1
        self.beta_parameters[pulled_arm, c_class, 1] = self.beta_parameters[pulled_arm, c_class, 1] + (1.0 - reward1)

        # update beta parameters associated with conversion rates on product 2, only if the first item has been bought
        if reward1 == 1:
            self.beta_cr2[c_class, promo, 0] = self.beta_cr2[c_class, promo, 0] + reward2
            self.beta_cr2[c_class, promo, 1] = self.beta_cr2[c_class, promo, 1] + (1.0 - reward2)


