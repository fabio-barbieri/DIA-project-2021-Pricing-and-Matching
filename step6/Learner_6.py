# THIS IS GOING TO BE THE SUPERCLASS OF THE THOMPSON SAMPLING AND GREEDY ALGORITHMS LEARNERS
import numpy as np
import config_6
from hungarian_algorithm_6 import hungarian_algorithm

np.random.seed(1234)

class Learner_6:
    def __init__(self, n_arms_1, n_arms_2):
        self.n_arms1 = n_arms_1
        self.n_arms2 = n_arms_2
        self.t1 = np.zeros((1, 4))
        self.t2 = np.zeros((4, 4))

        self.rewards_per_arm_1 = np.array([[] for _ in range(n_arms_1)])
        self.rewards_per_arm_2 = np.array([[] for _ in range(n_arms_2)])
        #self.collected_rewards = []

        self.m = np.array([config_6.TOT_CUSTOMERS // 4 for _ in range(4)])  # Non-informative prior
        self.s_2 = np.array([1.0, 1.0, 1.0, 1.0])
        self.expected_customers = np.random.normal(self.m, np.sqrt(self.s_2)).astype(int)  # initial number of expected customers per class, according to our prior
        self.matching = np.ones((4, 4)) / 16

        self.n_promos = np.array(config_6.PROMO_PROB * self.expected_customers)

        self.empirical_means_1 = np.zeros((n_arms_1, 4))
        self.confidence_1 = np.zeros((1, n_arms_1))
        self.empirical_means_2 = np.zeros((n_arms_2, 4, 4))
        self.confidence_2 = np.zeros((1, n_arms_2))


    def compute_posterior(self, x_bar):
        sigma_2 = config_6.SD_CUSTOMERS ** 2
        m = self.m
        s_2 = self.s_2

        self.m = (s_2 * x_bar + m * sigma_2) / (s_2 + sigma_2)
        self.s_2 = (s_2 * sigma_2) / (s_2 + sigma_2)
        self.expected_customers = np.random.normal(self.m, np.sqrt(self.s_2)).astype(int)

        # updating the array containing the number of promos, assigned by the marketing unit, for the next day
        self.n_promos = np.array([int(config_6.PROMO_PROB[i] * np.sum(self.expected_customers)) for i in range(1, 4)])
        self.n_promos = np.insert(self.n_promos, 0, np.sum(self.expected_customers) - np.sum(self.n_promos))

    def update(self, pulled_arm, reward1, reward2, c_class, promo):
        self.update_observations(pulled_arm, reward1, reward2)
        self.update_bounds(pulled_arm, reward1, reward2, c_class, promo)

    def update_bounds(self, pulled_arm, reward1, reward2, c_class, promo):
        self.t1[c_class] += 1  # Increment the counter of entered customers
        self.empirical_means_1[pulled_arm[0], c_class] = (self.empirical_means_1[pulled_arm[0], c_class] * (self.t1[c_class] - 1) + reward1) / self.t1[c_class]

        for arm in range(self.n_arms1):
            number_pulled = max(1, len(self.rewards_per_arm_1[arm]))
            self.confidence_1[arm] = (2 * np.log(np.sum(self.t1)) / number_pulled) ** 0.5

        if reward1 == 1:
            self.t2[c_class, promo] += 1
            self.empirical_means_2[pulled_arm[1], c_class, promo] = (self.empirical_means_2[pulled_arm[1], c_class, promo] * (self.t2[c_class, promo] - 1) + reward2) / self.t2[c_class, promo]

            for arm in range(self.n_arms2):
                number_pulled = max(1, len(self.rewards_per_arm_2[arm]))
                self.confidence_2[arm] = (2 * np.log(np.sum(self.t2)) / number_pulled) ** 0.5

    # NOTE: pulled arm here is a tuple (pulled_arm_1, pulled_arm_2)
    def update_observations(self, pulled_arm, reward1, reward2):
        self.rewards_per_arm_1[pulled_arm[0]].append(reward1)
        if reward1 == 1:
            self.rewards_per_arm_2[pulled_arm[1]].append(reward2)
        # self.collected_rewards.append(reward)  # List of N_EXPS arrays of shape (T, 4)

    def compute_matching(self):
        # hungarian algorithm starting matrix
        matrix, n_promo = self.build_matrix_optimistic()

        # matching is a binary matrix
        return hungarian_algorithm(matrix)  # returns (matching_mask, matching_value)

    def compute_matching_prob(self, matching_mask):
        customers_idxs = np.insert(np.cumsum(self.expected_customers), 0, 0)
        promo_idxs = np.insert(np.cumsum(self.n_promos, 0, 0))

        return np.array([np.sum(matching_mask[customers_idxs[i]:customers_idxs[i + 1], promo_idxs[j]:promo_idxs[j + 1]])
                        for i in range(5) for j in range(5)]).reshape((4, 4))

    def build_matrix_optimistic(self, idx1, idx2):
        matrix_dim = np.sum(self.expected_customers)

        matrix = np.zeros((4, 0))
        profit = np.array()
        n_promos = (config_6.PROMO_PROB[1:] * matrix_dim).astype(int)
        n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))

        # profit computed via ucb()...
        # n_promos = vettore con numero di promo assegnate per tipologia
        for i, n_promo in enumerate(n_promos):
            matrix = np.column_stack([matrix, np.repeat(profit[:, i].reshape(4, 1), n_promo, axis=1)])

        # n_customers = vettore con numero di clienti per classe
        matrix = np.repeat(matrix, self.expected_customers, axis=0)

        matrix = np.reshape(matrix, (matrix_dim, matrix_dim))

        return matrix

    def pull_arm(self):
        if np.sum(self.t1) < self.n_arms1 * self.n_arms2:
            idx1 = np.sum(self.t1) // self.n_arms2
            idx2 = np.sum(self.t1) % self.n_arms2
        else:
            upper_bound_1 = self.empirical_means_1 + self.confidence_1
            upper_bound_2 = self.empirical_means_2 + self.confidence_2

            for arm_1 in range(self.n_arms1):  # For every price_1
                for arm_2 in range(self.n_arms2):
                    self.build_matrix_optimistic(arm_1, arm_2)








                profit = 0
                for c_class in range(len(self.expected_customers)):  # For every customer class

                    exp_buyers_item1 = self.expected_customers[c_class] * upper_bound[arm]
                    margin1 = config_4.MARGINS_1[arm]
                    promo_assigment_prob = config_4.MATCHING_PROB[c_class, :] / self.expected_customers[
                        c_class] * np.sum(self.expected_customers)
                    margin2 = np.multiply(config_4.MARGINS_2, [
                        np.random.beta(self.beta_cr2[c_class, k, 0], self.beta_cr2[c_class, k, 1]) for k in
                        range(4)])

                    profit += exp_buyers_item1 * (margin1 + np.dot(promo_assigment_prob, margin2))

                profit /= np.sum(self.expected_customers)
                weighted_averages.append(profit)

            idx = np.argmax(weighted_averages)
        return idx1, idx2
