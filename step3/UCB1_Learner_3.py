from Learner_3 import Learner
import numpy as np
import config_3

class UCB1_Learner(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.t = 0

    def pull_arm(self):
        if self.t < self.n_arms:
            idx = self.t
        else:
            upper_bound = self.empirical_means + self.confidence

            weighted_averages = []
            for arm in range(self.n_arms):  # For every price_1
                profit = 0
                for c_class in range(len(config_3.NUM_CUSTOMERS)):  # For every customer class
                    exp_buyers_item1 = config_3.NUM_CUSTOMERS[c_class] * upper_bound[arm]
                    margin1 = config_3.MARGINS_1[arm]
                    promo_assigment_prob = config_3.MATCHING[c_class, :] / config_3.NUM_CUSTOMERS[c_class]
                    margin2 = np.multiply(config_3.MARGINS_2, config_3.CR2[c_class, :])

                    profit += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))            
                profit /= sum(config_3.NUM_CUSTOMERS)
                weighted_averages.append(profit)

            idx = np.argmax(weighted_averages)
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1 # Increment the counter of entered customers
        #self.collected_rewards = np.append(self.collected_rewards, reward)  # useless ???
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward) / self.t

        for arm in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[arm]))
            self.confidence[arm] = (2 * np.log(self.t) / number_pulled) ** 0.5
        
        self.rewards_per_arm[pulled_arm].append(reward)
