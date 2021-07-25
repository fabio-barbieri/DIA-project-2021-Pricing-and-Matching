import numpy as np
import config_6
from scipy.stats import truncnorm


class Environment_6():
    def __init__(self, num_customers, sd_customers, n_arms_1, n_arms_2, cr1, cr2):
        self.num_customers = num_customers
        self.sd_customers = sd_customers

        self.n_arms_1 = n_arms_1
        self.n_arms_2 = n_arms_2

        self.cr1 = cr1
        self.cr2 = cr2

    def customers(self):
        # extracting number of customer per class given a normal distribution
        clip_a = np.zeros(4)
        clip_b = self.num_customers * 5 / 2  
        a, b = (clip_a - self.num_customers) / self.sd_customers, (clip_b - self.num_customers) / self.sd_customers 
            
        customers = truncnorm.rvs(a, b, self.num_customers, self.sd_customers).astype(int)    

        tmp0 = np.zeros(shape=customers[0], dtype=int)
        tmp1 = np.ones(shape=customers[1], dtype=int)
        tmp2 = np.zeros(shape=customers[2], dtype=int) + 2
        tmp3 = np.zeros(shape=customers[3], dtype=int) + 3
        customer_arrivals = np.concatenate((tmp0, tmp1, tmp2, tmp3), axis=None)

        np.random.shuffle(customer_arrivals)

        current_daily_customers = np.array([len(tmp0), len(tmp1), len(tmp2), len(tmp3)])

        return customer_arrivals, current_daily_customers

    def round(self, pulled_arm, c_class, matching_prob, expected_customers):
        # reward by a single customer
        reward1 = np.random.binomial(1, self.cr1[pulled_arm[0]][c_class])

        # extracting promo assigned to the customer
        promo = np.random.choice([0, 1, 2, 3], p=matching_prob[c_class] / expected_customers[c_class] * np.sum(expected_customers))

        # reward in order to update cr2
        reward2 = np.random.binomial(1, self.cr2[pulled_arm[1]][c_class][promo])
        
        return reward1, reward2, promo