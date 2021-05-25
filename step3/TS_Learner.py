from Learner import *
import config


class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 4, 2))  # first param: price, second param: customer_class,
                                                        # third param: alpha, beta

    def pull_arm(self):
        # pull the arm that maximized the weighted average, over all the classes of users,
        # the conversion rate given by the beta distribution
        weighted_averages = []
        for arm in self.beta_parameters:
            conv_rate = 0
            for i, beta_distribution in enumerate(arm):
                conv_rate += config.num_customers[i] * np.random.beta(beta_distribution[0], beta_distribution[1])
            conv_rate /= sum(config.num_customers)
            weighted_averages.append(conv_rate)

        idx = np.argmax(weighted_averages)
        return idx

    def update(self, pulled_arm, rewards):
        self.t += 1
        self.update_observations(pulled_arm, rewards)
        self.beta_parameters[pulled_arm, :, 0] = self.beta_parameters[pulled_arm, :, 0] + rewards
        self.beta_parameters[pulled_arm, :, 1] = self.beta_parameters[pulled_arm, :, 1] + (config.num_customers - rewards)
