from CUSUM_UCB_Matching import *
import numpy as np
import config_8
from scipy.stats import truncnorm

np.random.seed(1234)

class Learner_8():
    def __init__(self, n_arms_1, n_arms_2, params):
        self.n_arms1 = n_arms_1
        self.n_arms2 = n_arms_2

        self.m = np.array([config_8.TOT_CUSTOMERS // 4 for _ in range(4)])  # Non-informative prior
        self.s_2 = np.array([1.0, 1.0, 1.0, 1.0])

        # initial number of expected customers per class, according to our prior 
        clip_a = np.zeros(4)
        clip_b = self.m * 5 / 2  
        a, b = (clip_a - self.m) / np.sqrt(self.s_2), (clip_b - self.m) / np.sqrt(self.s_2) 
        self.expected_customers = truncnorm.rvs(a, b, self.m, np.sqrt(self.s_2)).astype(int)

        # self.matching = np.ones((4, 4)) / 16

        self.n_promos = (config_8.PROMO_PROB[1 :] * np.sum(self.expected_customers)).astype(int)
        self.n_promos = np.insert(self.n_promos, 0, np.sum(self.expected_customers) - np.sum(self.n_promos))

        # self.rewards_per_arm_1 = [[[] for _ in range(4)] for _ in range(n_arms_1)]
        # self.rewards_per_arm_2 = [[[[] for _ in range(4)] for _ in range(4)] for _ in range(n_arms_2)]

        assert np.sum(self.expected_customers) == np.sum(self.n_promos)
        self.cusum_ucbs_per_arm = [CUSUM_UCB_Matching(n_rows=len(self.expected_customers), 
                                                      n_cols=len(self.n_promos), 
                                                      params=params) for _ in range(self.n_arms1 * self.n_arms2)]  # Arm = couple of p1, p2

    def pull_arm(self): # The selected arm is the couple p1, p2 that maximizes the value of the matching
        opt_value = -1
        for arm_1 in range(self.n_arms1):  # For every price_1
            for arm_2 in range(self.n_arms2):
                index = np.ravel_multi_index((arm_1, arm_2), dims=(self.n_arms1, self.n_arms2))  # Index to get the right cusum_ucb
                matching, mask = self.cusum_ucbs_per_arm[index].pull_cells(self.n_promos, self.expected_customers)
                value = np.sum(matching)
                if value > opt_value:
                    opt_mask = mask
                    opt_value = value
                    idx1 = arm_1
                    idx2 = arm_2

        matching_prob = self.compute_matching_prob(opt_mask)
        return matching_prob, (idx1, idx2)

    def compute_matching_prob(self, matching_mask):
        # inserted 0 at the start of idxs arrays, for ease of computation of the slices used in the return statement
        customers_idxs = np.insert(np.cumsum(self.expected_customers), 0, 0)
        promo_idxs = np.insert(np.cumsum(self.n_promos), 0, 0)

        matching_prob = np.array([np.sum(matching_mask[customers_idxs[i] : customers_idxs[i + 1], promo_idxs[j] : promo_idxs[j + 1]])
                                for i in range(4) for j in range(4)]).reshape((4, 4))

        # returning a (4, 4) matrix with P(class, promo) (aka MATCHING_PROB)
        return matching_prob / np.sum(self.expected_customers)

    def update(self, pulled_arm, c_class, promo, normalized_curr_profit):
        index = np.ravel_multi_index(pulled_arm, dims=(self.n_arms1, self.n_arms2))
        self.cusum_ucbs_per_arm[index].update(c_class, promo, normalized_curr_profit)

        for cusum_ucb in self.cusum_ucbs_per_arm:
            cusum_ucb.update_confidence()

    def compute_posterior(self, x_bar):
        sigma_2 = config_8.SD_CUSTOMERS ** 2
        m = self.m
        s_2 = self.s_2

        self.m = (s_2 * x_bar + m * sigma_2) / (s_2 + sigma_2)
        self.s_2 = (s_2 * sigma_2) / (s_2 + sigma_2)

        clip_a = np.zeros(4)
        clip_b = self.m * 5 / 2  
        a, b = (clip_a - self.m) / np.sqrt(self.s_2), (clip_b - self.m) / np.sqrt(self.s_2) 
        self.expected_customers = truncnorm.rvs(a, b, self.m, np.sqrt(self.s_2)).astype(int)

        matrix_dim = np.sum(self.expected_customers)
        self.n_promos = (config_8.PROMO_PROB[1 :] * matrix_dim).astype(int)
        self.n_promos = np.insert(self.n_promos, 0, matrix_dim - np.sum(self.n_promos))

    # def update_observations(self, pulled_arm, c_class, promo, reward1, reward2):
    #     self.rewards_per_arm_1[pulled_arm[0]][c_class].append(reward1)
    #     self.rewards_per_arm_2[pulled_arm[1]][c_class][promo].append(reward1 * reward2)

    
