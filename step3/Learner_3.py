class Learner_3:
    def __init__(self, n_arms, num_customers, margins_1, matching, margins_2, cr2):
        self.n_arms = n_arms
        self.num_customers = num_customers
        self.margins_1 = margins_1
        self.matching = matching
        self.margins_2 = margins_2
        self.cr2 = cr2

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards.append(reward)  # List of N_EXPS arrays of shape (T, 4)
