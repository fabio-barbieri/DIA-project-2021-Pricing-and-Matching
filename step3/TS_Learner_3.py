import numpy as np
from Learner_3 import *

np.random.seed(1234)


class TS_Learner_3(Learner_3):
    
    def __init__(self, n_arms, num_customers, margins_1, matching, margins_2, cr2):
        super().__init__(n_arms, num_customers, margins_1, matching, margins_2, cr2)
        # Array used to store the values of parameters alpha and beta for the
        # beta distributions related at each class of customers, for each possible arm
        self.beta_parameters = np.ones((n_arms, 4, 2))  # n_arms = num of item 1 prices
                                                        # 4 = num of customer classes
                                                        # 2 = amount of parameters for the beta distribution (alpa, beta)
        
    def profit(self, i, arm):
        cr1 = np.random.beta(arm[:, 0], arm[:, 1])                                 # 4x1
        margin1 = self.margins_1[i]                                                # 1x1
        cr2 = self.cr2                                                             # 4x4
        matching_prob = self.matching / np.expand_dims(self.num_customers, axis=1) # 4x4
        margins2 = self.margins_2                                                  # 4x1

        a = cr1 * (margin1 + np.dot(cr2 * matching_prob, margins2)) #   4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                    # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                    # = 4x1 * (1x1 + 4x1) = 
                                                                    # = 4x1 * 4x1 = 
                                                                    # = 4x1
        return np.dot(a, self.num_customers)

    def pull_arm(self):
        # Pull the arm that maximizes the profit w.r.t. all the classes of customers and the beta distributions

        profits = [self.profit(i, arm) for i, arm in enumerate(self.beta_parameters)]
        return np.argmax(profits)

        # Pull the arm that maximizes the weighted average of the conv rates over all
        # the classes of customers w.r.t. the beta distribution

        #for i, arm in enumerate(self.beta_parameters):  # For every price_1
        #    cr = 0
        #    for j, params in enumerate(arm):  # For every customer class
        #        exp_buyers_item1 = self.num_customers[j] * np.random.beta(params[0], params[1])
        #        margin1 = self.margins_1[i]
        #        promo_assigment_prob = self.matching[j, :] / self.num_customers[j]
        #        margin2 = np.multiply(self.margins_2, self.cr2[j, :])
        #
        #        cr += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))            
        #    cr /= sum(self.num_customers)
        #    weighted_averages.append(cr)
        #
        #idx = np.argmax(weighted_averages)
        #return idx

    def update(self, pulled_arm, reward, c_class):
        # self.t += 1
        self.update_observations(pulled_arm, reward)
        # Update the parameters of the betas according to the rewards and considering that the average num
        # of customers per class must be considered
        self.beta_parameters[pulled_arm, c_class, 0] = self.beta_parameters[pulled_arm, c_class, 0] + reward
        self.beta_parameters[pulled_arm, c_class, 1] = self.beta_parameters[pulled_arm, c_class, 1] + (1.0 - reward)
