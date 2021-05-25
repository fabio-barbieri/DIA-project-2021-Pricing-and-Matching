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
        # Pull the arm that maximizes the weightes average of the conv rates over all 
        # the classes of customers w.r.t. the beta distribution
        weighted_averages = []
        for arm in self.beta_parameters:
            cr = 0
            for i, params in enumerate(arm):
                cr += config.NUM_CUSTOMERS[i] * np.random.beta(params[0], params[1])
            cr /= sum(config.NUM_CUSTOMERS)
            weighted_averages.append(cr)

        idx = np.argmax(weighted_averages)
        return idx

    def update(self, pulled_arm, rewards):
        self.t += 1
        self.update_observations(pulled_arm, rewards)
        # Update the parameters of the betas according to the rewards and considering that the average num
        # of customers per class must be condidered
        self.beta_parameters[pulled_arm, :, 0] = self.beta_parameters[pulled_arm, :, 0] + rewards
        self.beta_parameters[pulled_arm, :, 1] = self.beta_parameters[pulled_arm, :, 1] + (config.NUM_CUSTOMERS - rewards)
