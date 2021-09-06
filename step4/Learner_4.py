import numpy as np
from scipy.stats import truncnorm

np.random.seed(1234)

class Learner_4:
    def __init__(self, n_arms, tot_customers, sd_customers, margins_1, matching_prob, margins_2):
        self.n_arms = n_arms
        self.tot_customers = tot_customers
        self.sd_customers = sd_customers 
        self.margins_1 = margins_1
        self.matching_prob = matching_prob
        self.margins_2 = margins_2

        self.m = np.array([self.tot_customers // 4 for _ in range(4)])  # Non-informative prior
        self.s_2 = np.array([1.0, 1.0, 1.0, 1.0])
        
        clip_a = np.zeros(4)
        clip_b = self.m * 5 / 2  
        a, b = (clip_a - self.m) / np.sqrt(self.s_2), (clip_b - self.m) / np.sqrt(self.s_2) 
        self.expected_customers = truncnorm.rvs(a, b, self.m, np.sqrt(self.s_2)).astype(int) # initial number of expected customers
                                                                                             # per class, according to our prior 

    def compute_posterior(self, x_bar):
        sigma_2 = self.sd_customers ** 2
        m = self.m
        s_2 = self.s_2

        self.m = (s_2 * x_bar + m * sigma_2) / (s_2 + sigma_2)
        self.s_2 = (s_2 * sigma_2) / (s_2 + sigma_2)
        
        clip_a = np.zeros(4)
        clip_b = self.m * 5 / 2 
        a, b = (clip_a - self.m) / np.sqrt(self.s_2), (clip_b - self.m) / np.sqrt(self.s_2) 
        self.expected_customers = truncnorm.rvs(a, b, self.m, np.sqrt(self.s_2)).astype(int)