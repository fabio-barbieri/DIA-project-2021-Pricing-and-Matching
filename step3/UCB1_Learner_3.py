import numpy as np
from Learner_3 import *

class UCB1_Learner_3(Learner_3):

    def __init__(self, n_arms, num_customers, margins_1, matching, margins_2, cr2):
        super().__init__(n_arms, num_customers, margins_1, matching, margins_2, cr2)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.t = 0

    def profit(self, i, upper_bound):
        cr1 = upper_bound[i]                                                       # 4x1
        margins1 = self.margins_1[i]                                               # 1x1
        cr2 = self.cr2                                                             # 4x4
        matching_prob = self.matching / np.expand_dims(self.num_customers, axis=1) # 4x4
        margins2 = self.margins_2                                                  # 4x1

        a = cr1 * (margins1 + np.dot(cr2 * matching_prob, margins2)) #   4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                     # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                     # = 4x1 * (1x1 + 4x1) = 
                                                                     # = 4x1 * 4x1 = 
                                                                     # = 4x1
        return np.dot(a, self.num_customers)

    def pull_arm(self):
        if self.t < self.n_arms:
            idx = self.t
        else:
            upper_bound = self.empirical_means + self.confidence

            profits = [self.profit(i, upper_bound) for i in range(self.n_arms)]
            idx = np.argmax(profits)
        
        return idx 

            #weighted_averages = []
            #for arm in range(self.n_arms):  # For every price_1
            #    profit = 0
            #    for c_class in range(len(self.num_customers)):  # For every customer class
            #        exp_buyers_item1 = self.num_customers[c_class] * upper_bound[arm]
            #        margin1 = self.num_margins_1[arm]
            #        promo_assigment_prob = self.num_matching[c_class, :] / self.num_customers[c_class]
            #        margin2 = np.multiply(self.num_margins_2, self.cr2[c_class, :])
            #
            #        profit += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))            
            #    profit /= sum(self.num_customers)
            #    weighted_averages.append(profit)
            #
            #idx = np.argmax(weighted_averages)
        #return idx

    def update(self, pulled_arm, reward):
        self.t += 1 # Increment the counter of entered customers
        #self.collected_rewards = np.append(self.collected_rewards, reward)  # useless ???
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward) / self.t

        for arm in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[arm]))
            self.confidence[arm] = (2 * np.log(self.t) / number_pulled) ** 0.5
        
        self.rewards_per_arm[pulled_arm].append(reward)
