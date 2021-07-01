import numpy as np
import config


class Environment():
    def __init__(self, n_arms, cr1, cr2):
        self.n_arms = n_arms
        self.cr1 = cr1
        self.cr2 = cr2

    def round(self, pulled_arm, c_class, current_daily_customers):
        # reward by a single customer
        reward1 = np.random.binomial(1, self.cr1[pulled_arm][c_class])

        # extracting promo assigned to the customer
        promo = np.random.choice([0, 1, 2, 3], p=config.PROMO_PROB)

        # reward in order to update cr2
        reward2 = np.random.binomial(1, self.cr2[c_class][promo])

        return reward1, reward2, promo
