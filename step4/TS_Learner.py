import numpy as np
from Learner import *
import config


class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # Array used to store the values of parameters alpha and beta for the
        # beta distributions related at each class of customers, for each possible arm
        self.beta_parameters = np.ones((n_arms, 4, 2))  # n_arms = num of item 1 prices
                                                        # 4 = num of customer classes
                                                        # 2 = amount of parameters for the beta distribution (alpa, beta)
        self.expected_customers = np.array([25, 25, 25, 25])
        self.beta_cr2 = np.ones(shape=(4, 4, 2))

    def pull_arm(self):
        # Pull the arm that maximizes the weighted average of the conv rates over all
        # the classes of customers w.r.t. the beta distribution
        weighted_averages = []
        for i, arm in enumerate(self.beta_parameters):  # For every price_1
            cr = 0
            for j, params in enumerate(arm):  # For every customer class
                exp_buyers_item1 = self.expected_customers[j] * np.random.beta(params[0], params[1])
                margin1 = config.MARGINS_1[i]
                promo_assigment_prob = config.MATCHING_PROB[j, :] / self.expected_customers[j] * np.sum(self.expected_customers)
                margin2 = np.multiply(config.MARGINS_2, [np.random.beta(self.beta_cr2[j, k, 0], self.beta_cr2[j, k, 1]) for k in range(4)])

                cr += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))            
            cr /= np.sum(self.expected_customers)
            weighted_averages.append(cr)

        idx = np.argmax(weighted_averages)
        return idx

    def update(self, pulled_arm, reward1, reward2, c_class, promo):
        # self.t += 1
        self.update_observations(pulled_arm, reward1)
        # Update the parameters of the betas according to the rewards and considering that the average num
        # of customers per class must be considered
        self.beta_parameters[pulled_arm, c_class, 0] = self.beta_parameters[pulled_arm, c_class, 0] + reward1
        self.beta_parameters[pulled_arm, c_class, 1] = self.beta_parameters[pulled_arm, c_class, 1] + (1.0 - reward1)

        # update beta parameters associated with conversion rates on product 2, only if the first item has been bought
        if reward1 == 1:
            self.beta_cr2[c_class, promo, 0] = self.beta_cr2[c_class, promo, 0] + reward2
            self.beta_cr2[c_class, promo, 1] = self.beta_cr2[c_class, promo, 1] + (1.0 - reward2)

    def update_expected_customers(self, current_daily_customers, t):
        self.expected_customers = (self.expected_customers * (t - 1) + current_daily_customers) / t

