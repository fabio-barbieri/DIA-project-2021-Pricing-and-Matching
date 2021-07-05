# THIS IS GOING TO BE THE SUPERCLASS OF THE THOMPSON SAMPLING AND GREEDY ALGORITHMS LEARNERS
import numpy as np
import config_5

np.random.seed(1234)

class Learner_5:
    def __init__(self):
        #self.rewards_per_arm = [[] for _ in range(n_arms)]
        #self.collected_rewards = []
        self.m = np.array([config_5.TOT_CUSTOMERS // 4 for _ in range(4)])  # Non-informative prior
        self.s_2 = np.array([1.0, 1.0, 1.0, 1.0])
        self.expected_customers = np.random.normal(self.m, np.sqrt(self.s_2)).astype(int)  # initial number of expected customers
                                                                                    # per class, according to our prior
        self.beta_cr1 = np.ones((4, 2))
        self.beta_cr2 = np.ones((4, 4, 2))

    def update_betas(self, reward1, reward2, c_class, promo):
        # Update the parameters of the betas according to the rewards and considering that the average num
        # of customers per class must be considered
        self.beta_cr1[c_class, 0] = self.beta_cr1[c_class, 0] + reward1
        self.beta_cr1[c_class, 1] = self.beta_cr1[c_class, 1] + (1.0 - reward1)

        # update beta parameters associated with conversion rates on product 2, only if the first item has been bought
        if reward1 == 1:
            self.beta_cr2[c_class, promo, 0] = self.beta_cr2[c_class, promo, 0] + reward2
            self.beta_cr2[c_class, promo, 1] = self.beta_cr2[c_class, promo, 1] + (1.0 - reward2)


    def compute_posterior(self, x_bar):
        sigma_2 = config_5.SD_CUSTOMERS ** 2
        m = self.m
        s_2 = self.s_2

        self.m = (s_2 * x_bar + m * sigma_2) / (s_2 + sigma_2)
        self.s_2 = (s_2 * sigma_2) / (s_2 + sigma_2)
        self.expected_customers = np.random.normal(self.m, np.sqrt(self.s_2)).astype(int)



    def compute_matching(self):
        # hungarian algorithm starting matrix
        matrix = self.build_matrix()

        return None

    def build_matrix(self):
        matrix_dim = np.sum(self.expected_customers)
        matrix = np.array([])

        # array containing the number of promos, assigned by the marketing unit
        n_promos = np.array([int(config_5.PROMO_PROB[i] * matrix_dim) for i in range(1, 4)])
        n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))

        customer0_row = np.array([])
        customer1_row = np.array([])
        customer2_row = np.array([])
        customer3_row = np.array([])

        for i, n_promo in enumerate(n_promos):
            customer0_row = np.append(customer0_row, [
                config_5.MARGIN_1 * np.random.beta(self.beta_cr1[0, 0], self.beta_cr1[0, 1]) + config_5.MARGINS_2[
                    i] * np.random.beta(self.beta_cr2[i, 0, 0], self.beta_cr2[i, 0, 1]) for _ in range(n_promo)])
            customer1_row = np.append(customer1_row, [
                config_5.MARGIN_1 * np.random.beta(self.beta_cr1[1, 0], self.beta_cr1[1, 1]) + config_5.MARGINS_2[
                    i] * np.random.beta(self.beta_cr2[i, 1, 0], self.beta_cr2[i, 1, 1]) for _ in range(n_promo)])
            customer2_row = np.append(customer2_row, [
                config_5.MARGIN_1 * np.random.beta(self.beta_cr1[2, 0], self.beta_cr1[2, 1]) + config_5.MARGINS_2[
                    i] * np.random.beta(self.beta_cr2[i, 2, 0], self.beta_cr2[i, 2, 1]) for _ in range(n_promo)])
            customer3_row = np.append(customer3_row, [
                config_5.MARGIN_1 * np.random.beta(self.beta_cr1[3, 0], self.beta_cr1[3, 1]) + config_5.MARGINS_2[
                    i] * np.random.beta(self.beta_cr2[i, 3, 0], self.beta_cr2[i, 3, 1]) for _ in range(n_promo)])

        for i, num in enumerate(self.expected_customers):
            for _ in range(num):
                if i == 0:
                    matrix = np.concatenate((matrix, customer0_row), axis=0)
                elif i == 1:
                    matrix = np.concatenate((matrix, customer1_row), axis=0)
                elif i == 2:
                    matrix = np.concatenate((matrix, customer2_row), axis=0)
                else:
                    matrix = np.concatenate((matrix, customer3_row), axis=0)

        matrix = -1 * matrix  # to face the maximization problem with hungarian algorithm

        return np.reshape(matrix, (matrix_dim, matrix_dim))


if __name__ == '__main__':
    l = Learner_5()
    l.compute_matching()





