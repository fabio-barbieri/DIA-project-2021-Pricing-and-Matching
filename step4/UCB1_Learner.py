from Learner import Learner
import numpy as np
import config

class UCB1_Learner(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.expected_customers = np.array([25, 25, 25, 25])
        self.beta_cr2 = np.ones((4, 4, 2))
        self.t = 0

    def pull_arm(self):
        if self.t < self.n_arms:
            idx = self.t
        else:
            upper_bound = self.empirical_means + self.confidence

            weighted_averages = []
            for arm in range(self.n_arms):  # For every price_1
                profit = 0
                for c_class in range(len(self.expected_customers)):  # For every customer class
                    exp_buyers_item1 = self.expected_customers[c_class] * upper_bound[arm]
                    margin1 = config.MARGINS_1[arm]
                    promo_assigment_prob = config.MATCHING_PROB[c_class, :] / self.expected_customers[c_class] * np.sum(self.expected_customers)
                    margin2 = np.multiply(config.MARGINS_2, [np.random.beta(self.beta_cr2[c_class, k, 0], self.beta_cr2[c_class, k, 1]) for k in range(4)])

                    profit += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))            
                profit /= np.sum(self.expected_customers)
                weighted_averages.append(profit)

            idx = np.argmax(weighted_averages)
        return idx

    def update(self, pulled_arm, reward1, reward2, c_class, promo):
        self.t += 1 # Increment the counter of entered customers
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward1) / self.t

        for arm in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[arm]))
            self.confidence[arm] = (2 * np.log(self.t) / number_pulled) ** 0.5

        # update beta parameters associated with conversion rates on product 2, if the first item has been bought
        if reward1 == 1:
            self.beta_cr2[c_class, promo, 0] = self.beta_cr2[c_class, promo, 0] + reward2
            self.beta_cr2[c_class, promo, 1] = self.beta_cr2[c_class, promo, 1] + (1.0 - reward2)

    def update_expected_customers(self, current_daily_customers, t):
        self.expected_customers = (self.expected_customers * (t - 1) + current_daily_customers) / t
