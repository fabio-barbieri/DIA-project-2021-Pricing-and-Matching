from numpy import poly1d, promote_types
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

    def pull_arm(self):
        # Pull the arm that maximizes the weighted average of the conv rates over all
        # the classes of customers w.r.t. the beta distribution
        weighted_averages = []
        for i, arm in enumerate(self.beta_parameters):  # For every price_1
            cr = 0
            for j, params in enumerate(arm):  # For every customer class
                exp_buyers_item1 = config.NUM_CUSTOMERS[j] * np.random.beta(params[0], params[1])
                margin1 = config.MARGINS_1[i]
                promo_assigment_prob = config.MATCHING[j, :] / config.NUM_CUSTOMERS[j]
                margin2 = np.multiply(config.MARGINS_2, config.CR2[j, :])

                cr += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))            
            cr /= sum(config.NUM_CUSTOMERS)
            weighted_averages.append(cr)

        idx = np.argmax(weighted_averages)
        return idx

    def update(self, pulled_arm, reward, c_class):
        # self.t += 1
        self.update_observations(pulled_arm, reward)
        # Update the parameters of the betas according to the rewards and considering that the average num
        # of customers per class must be considered
        self.beta_parameters[pulled_arm, c_class, 0] = self.beta_parameters[pulled_arm, c_class, 0] + reward
        self.beta_parameters[pulled_arm, c_class, 1] = self.beta_parameters[pulled_arm, c_class, 1] + (1.0 - reward)
