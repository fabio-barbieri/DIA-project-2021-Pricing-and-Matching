from Learner_4 import *
import numpy as np


class UCB1_Learner_4(Learner_4):

    def __init__(self, n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2):
        super().__init__(n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.beta_cr2 = np.ones((4, 4, 2))
        self.t = 0

    def profit(self, i, upper_bound):
        cr1 = upper_bound[i]                                                                                                   # 4x1
        margins1 = self.margins_1[i]                                                                                           # 1x1
        cr2 = np.random.beta(self.beta_cr2[:, :, 0], self.beta_cr2[:, :, 1])                                                   # 4x4
        matching_prob = self.matching_prob / np.expand_dims(self.expected_customers, axis=1) * np.sum(self.expected_customers) # 4x4
        margins2 = self.margins_2                                                                                              # 4x1

        a = cr1 * (margins1 + np.dot(cr2 * matching_prob, margins2)) #   4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                     # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                     # = 4x1 * (1x1 + 4x1) = 
                                                                     # = 4x1 * 4x1 = 
                                                                     # = 4x1
        return np.dot(a, self.expected_customers)

    def pull_arm(self):
        if self.t < self.n_arms:
            idx = self.t
        else:
            upper_bound = self.empirical_means + self.confidence

            profits = [self.profit(i, upper_bound) for i in range(self.n_arms)]
            idx = np.argmax(profits)
        
        return idx

        #    weighted_averages = []
        #    for arm in range(self.n_arms):  # For every price_1
        #        profit = 0
        #        for c_class in range(len(self.expected_customers)):  # For every customer class
        #
        #            exp_buyers_item1 = self.expected_customers[c_class] * upper_bound[arm]
        #            margin1 = self.margins_1[arm]
        #            promo_assigment_prob = self.matching_prob[c_class, :] / self.expected_customers[c_class] * np.sum(self.expected_customers)
        #            margin2 = np.multiply(self.margins_2, [np.random.beta(self.beta_cr2[c_class, k, 0], self.beta_cr2[c_class, k, 1]) for k in range(4)])
        #
        #            profit += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))
        #
        #        profit /= np.sum(self.expected_customers)
        #        weighted_averages.append(profit)
        #
        #    idx = np.argmax(weighted_averages)
        #return idx

    def update(self, pulled_arm, reward1, reward2, c_class, promo):
        self.t += 1  # Increment the counter of entered customers
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward1) / self.t

        for arm in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[arm]))
            self.confidence[arm] = (2 * np.log(self.t) / number_pulled) ** 0.5

        # update beta parameters associated with conversion rates on product 2, if the first item has been bought
        if reward1 == 1:
            self.beta_cr2[c_class, promo, 0] = self.beta_cr2[c_class, promo, 0] + reward2
            self.beta_cr2[c_class, promo, 1] = self.beta_cr2[c_class, promo, 1] + (1.0 - reward2)

        self.rewards_per_arm[pulled_arm].append(reward1)

