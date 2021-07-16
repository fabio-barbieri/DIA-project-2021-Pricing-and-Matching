# THIS IS GOING TO BE THE SUPERCLASS OF THE THOMPSON SAMPLING AND GREEDY ALGORITHMS LEARNERS
import numpy as np
import config_5
from hungarian_algorithm import hungarian_algorithm

np.random.seed(1234)

class Learner_5:
    def __init__(self):
        self.m = np.array([config_5.TOT_CUSTOMERS // 4 for _ in range(4)])  # Non-informative prior
        self.s_2 = np.array([1.0, 1.0, 1.0, 1.0])
        self.expected_customers = np.random.normal(self.m, np.sqrt(self.s_2)).astype(int)  # initial number of expected customers
                                                                                           # per class, according to our prior                                                               
        self.beta_cr1 = np.ones((4, 2))
        self.beta_cr2 = np.ones((4, 4, 2))


    def build_matrix(self):
        matrix_dim = np.sum(self.expected_customers)

        matrix = np.zeros((4, 0))
        sampled_cr1 = np.random.beta(self.beta_cr1[:, 0], self.beta_cr1[:, 1])
        sampled_cr2 = np.random.beta(self.beta_cr2[:, :, 0], self.beta_cr2[:, :, 1])
        profit = sampled_cr1 * (config_5.MARGINS_1 + sampled_cr2 * config_5.MARGINS_2)

        # First set integers p1, p2, p3 and the remaining are p0 
        n_promos = (config_5.PROMO_PROB[1 :] * matrix_dim).astype(int)
        n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))

        # repeat columns
        matrix = np.column_stack([matrix, np.repeat(profit, n_promos, axis=1)])

        # repeat rows
        matrix = np.repeat(matrix, self.expected_customers, axis=0)

        return matrix


    def compute_matching(self):
        # hungarian algorithm starting matrix
        matrix = self.build_matrix()

        return hungarian_algorithm(matrix)


    def compute_matching_prob(self, matching_mask):
        # inserted 0 at the start of idxs arrays, for ease of computation of the slices used in the return statement
        customers_idxs = np.insert(np.cumsum(self.expected_customers), 0, 0)
        promo_idxs = np.insert(np.cumsum(self.n_promos), 0, 0)

        matching_prob = np.array([np.sum(matching_mask[customers_idxs[i] : customers_idxs[i + 1], promo_idxs[j] : promo_idxs[j + 1]])
                                 for i in range(4) for j in range(4)]).reshape((4, 4))

        # returning a (4, 4) matrix with P(class, promo) (aka MATCHING_PROB)
        return matching_prob / np.sum(self.expected_customers)


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


if __name__ == '__main__':
    l = Learner_5()
    l.compute_matching()





