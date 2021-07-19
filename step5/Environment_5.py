import numpy as np


class Environment_5():
    def __init__(self, cr1, cr2, num_customers, sd_customers):
        self.cr1 = cr1
        self.cr2 = cr2
        self.num_customers = num_customers
        self.sd_customers = sd_customers

    def customers(self):
        # extracting number of customer per class given a normal distribution
        tmp0 = np.zeros(shape=int(np.random.normal(self.num_customers[0], self.sd_customers[0])), dtype=int)
        tmp1 = np.ones(shape=int(np.random.normal(self.num_customers[1], self.sd_customers[1])), dtype=int)
        tmp2 = np.zeros(shape=int(np.random.normal(self.num_customers[2], self.sd_customers[2])), dtype=int) + 2
        tmp3 = np.zeros(shape=int(np.random.normal(self.num_customers[3], self.sd_customers[3])), dtype=int) + 3
        customer_arrivals = np.array([], dtype=int)
        customer_arrivals = np.concatenate((customer_arrivals, tmp0), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp1), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp2), axis=None)
        customer_arrivals = np.concatenate((customer_arrivals, tmp3), axis=None)

        current_daily_customers = np.array([len(tmp0), len(tmp1), len(tmp2), len(tmp3)])

        return customer_arrivals, current_daily_customers


    def round(self, c_class, matching_prob, expected_customers):
        # reward by a single customer
        reward1 = np.random.binomial(1, self.cr1[c_class])

        # extracting promo assigned to the customer
        promo = np.random.choice([0, 1, 2, 3], p=matching_prob[c_class] / expected_customers[c_class] * np.sum(expected_customers))

        # reward in order to update cr2
        reward2 = np.random.binomial(1, self.cr2[c_class][promo])

        return reward1, reward2, promo
