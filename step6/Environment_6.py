import numpy as np
import config_6


class Environment_6():
    def __init__(self, n_arms_1, n_arms_2, cr1, cr2):
        self.n_arms_1 = n_arms_1
        self.n_arms_2 = n_arms_2
        self.cr1 = cr1
        self.cr2 = cr2

    def customers(self):
        # extracting number of customer per class given a normal distribution
        tmp0 = np.zeros(shape=int(np.random.normal(config_6.NUM_CUSTOMERS[0], config_6.SD_CUSTOMERS[0])), dtype=int)
        tmp1 = np.ones(shape=int(np.random.normal(config_6.NUM_CUSTOMERS[1], config_6.SD_CUSTOMERS[1])), dtype=int)
        tmp2 = np.zeros(shape=int(np.random.normal(config_6.NUM_CUSTOMERS[2], config_6.SD_CUSTOMERS[2])), dtype=int) + 2
        tmp3 = np.zeros(shape=int(np.random.normal(config_6.NUM_CUSTOMERS[3], config_6.SD_CUSTOMERS[3])), dtype=int) + 3
        customer_arrivals = np.array([], dtype=int)
        customer_arrivals = np.concatenate((customer_arrivals, tmp0), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp1), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp2), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp3), axis=None)

        np.random.shuffle(customer_arrivals)

        current_daily_customers = np.array([len(tmp0), len(tmp1), len(tmp2), len(tmp3)])

        return customer_arrivals, current_daily_customers

    def round(self, pulled_arm, c_class):
        # reward by a single customer
        reward1 = np.random.binomial(1, self.cr1[pulled_arm[0]][c_class])

        # extracting promo assigned to the customer
        promo = np.random.choice([0, 1, 2, 3], p=config_6.MATCHING_PROB[c_class] / config_6.NUM_CUSTOMERS[c_class] * config_6.TOT_CUSTOMERS)

        # reward in order to update cr2
        reward2 = np.random.binomial(1, self.cr2[pulled_arm[1]][c_class][promo])

        return reward1, reward2, promo
