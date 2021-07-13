# THIS IS GOING TO BE THE SUPERCLASS OF THE THOMPSON SAMPLING AND GREEDY ALGORITHMS LEARNERS
import numpy as np
import config_6
from hungarian_algorithm_6 import hungarian_algorithm

np.random.seed(1234)

class Learner_6:
    def __init__(self, n_arms_1, n_arms_2):
        self.n_arms1 = n_arms_1
        self.n_arms2 = n_arms_2
        self.t1 = np.zeros((4, 1), dtype=int)
        self.t2 = np.zeros((4, 4), dtype=int)

        self.n_pulled_arm_1 = np.zeros((n_arms_1, 1))
        self.n_pulled_arm_2 = np.zeros((n_arms_2, 1))
        self.n_pulled_couple = np.zeros((n_arms_1, n_arms_2))

        self.m = np.array([config_6.TOT_CUSTOMERS // 4 for _ in range(4)])  # Non-informative prior
        self.s_2 = np.array([1.0, 1.0, 1.0, 1.0])
        self.expected_customers = np.random.normal(self.m, np.sqrt(self.s_2)).astype(int)  # initial number of expected customers per class, according to our prior
        self.matching = np.ones((4, 4)) / 16

        self.n_promos = np.array([int(config_6.PROMO_PROB[i] * np.sum(self.expected_customers)) for i in range(1, 4)])
        self.n_promos = np.insert(self.n_promos, 0, np.sum(self.expected_customers) - np.sum(self.n_promos))

        self.empirical_means_1 = np.zeros((n_arms_1, 4))
        self.confidence_1 = np.zeros((n_arms_1, 1))
        self.empirical_means_2 = np.zeros((n_arms_2, 4, 4))
        self.confidence_2 = np.zeros((n_arms_2, 1, 1))


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
            number_pulled = max(1, self.n_pulled_arm_1[arm])
            self.confidence_1[arm, 0] = (2 * np.log(np.sum(self.t1)) / number_pulled) ** 0.5

        if reward1 == 1:
            self.t2[c_class, promo] += 1
            self.empirical_means_2[pulled_arm[1], c_class, promo] = (self.empirical_means_2[pulled_arm[1], c_class, promo] * (self.t2[c_class, promo] - 1) + reward2) / self.t2[c_class, promo]

            for arm in range(self.n_arms2):
                number_pulled = max(1, self.n_pulled_arm_2[arm])
                self.confidence_2[arm, 0, 0] = (2 * np.log(np.sum(self.t2)) / number_pulled) ** 0.5

    # NOTE: pulled arm here is a tuple (pulled_arm_1, pulled_arm_2)
    def update_observations(self, pulled_arm, reward1, reward2):
        self.n_pulled_arm_1[pulled_arm[0]] += 1
        self.n_pulled_arm_2[pulled_arm[1]] += 1

        self.n_pulled_couple[pulled_arm[0], pulled_arm[1]] += 1

    def compute_matching_prob(self, matching_mask):

        # inserted 0 at the start of idxs arrays, for ease of computation of the slices used in the return statement
        customers_idxs = np.insert(np.cumsum(self.expected_customers), 0, 0)
        promo_idxs = np.insert(np.cumsum(self.n_promos), 0, 0)

        matching_prob = np.array([np.sum(matching_mask[customers_idxs[i]:customers_idxs[i + 1], promo_idxs[j]:promo_idxs[j + 1]])
                                 for i in range(4) for j in range(4)]).reshape((4, 4))

        # returning a (4, 4) matrix with P(class, promo) (aka MATCHING_PROB)
        return matching_prob / np.sum(self.expected_customers)

    def build_matrix_optimistic(self, idx1, idx2, ub_cr1, ub_cr2):
        matrix_dim = np.sum(self.expected_customers)

        matrix = np.zeros((4, 0))
        profit = np.multiply(ub_cr1[idx1].reshape((4, 1)), config_6.MARGINS_1[idx1] + np.multiply(ub_cr2[idx2],
                             config_6.MARGINS_2[idx2]))

        n_promos = (config_6.PROMO_PROB[1:] * matrix_dim).astype(int)
        n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))

        # repeat columns
        matrix = np.column_stack([matrix, np.repeat(profit, n_promos, axis=1)])

        # repeat rows
        matrix = np.repeat(matrix, self.expected_customers, axis=0)

        return matrix

    def pull_arm(self):
        # bounds on the conversion rates that will be used to compute the matrix for the matching
        upper_bound_1 = self.empirical_means_1 + self.confidence_1
        upper_bound_2 = self.empirical_means_2 + self.confidence_2

        if np.sum(self.t1) < self.n_arms1 * self.n_arms2:
            idx1 = np.sum(self.t1) // self.n_arms2
            idx2 = np.sum(self.t1) % self.n_arms2

            # building matrix in order to be able to compute total daily reward with these arms pulled
            matching, value = hungarian_algorithm(self.build_matrix_optimistic(idx1, idx2, upper_bound_1, upper_bound_2))
            matching_prob = self.compute_matching_prob(matching)
        else:
            opt_value = -1
            for arm_1 in range(self.n_arms1):  # For every price_1
                for arm_2 in range(self.n_arms2):
                    matching, value = hungarian_algorithm(self.build_matrix_optimistic(arm_1, arm_2, upper_bound_1, upper_bound_2))
                    if value > opt_value:
                        opt_matching = matching
                        opt_value = value
                        idx1 = arm_1
                        idx2 = arm_2

            matching_prob = self.compute_matching_prob(opt_matching)

        return matching_prob, (idx1, idx2)

