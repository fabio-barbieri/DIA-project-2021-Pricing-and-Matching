import numpy as np
from CUSUM import CUSUM
from scipy.optimize import linear_sum_assignment

np.random.seed(1234)

class CUSUM_UCB_Matching():
    def __init__(self, n_rows, n_cols, M=100, eps=0.05, h=20, alpha=0.01):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_cells = self.n_rows * self.n_cols # num of cells in the matching adjacency matrix

        self.change_detection = [CUSUM(M, eps, h) for _ in range(self.n_cells)]
        self.valid_rewards_per_cell = [[] for _ in range(self.n_cells)]
        self.detections = [[] for _ in range(self.n_cells)]  # to keep track of how many times a warning has been raised for a specific cell
        self.alpha = alpha

    
    def pull_cells(self):

        def compute_matching(matrix):
            rows, cols = linear_sum_assignment(matrix, maximize=True)
            matching_mask = np.zeros(matrix.shape, dtype=int)
            matching_mask[rows, cols] = 1
            return rows, cols, matching_mask * matrix, matching_mask

        if np.random.binomial(1, 1 - self.alpha):  # we are going to execute this block of code with probability 1-alpha (exploitation)
            upper_conf = self.empirical_means + self.confidence
            upper_conf[np.isinf(upper_conf)] = 1e3
            # returning --> row_ind, col_ind, matching, matching_mask
            rows, cols, matching, mask = compute_matching(upper_conf.reshape(self.n_rows, self.n_cols))

        else:  # with probability alpha we get a random matching, pulling, in this way, random arms (exploration)
            random_costs = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            # returning --> row_ind, col_ind, matching, matching_mask RANDOM!!
            rows, cols, matching, mask = compute_matching(random_costs)
        
        return rows, cols, matching, mask

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for pulled_arm, reward in zip(pulled_arm_flat, rewards):
            if self.change_detection[pulled_arm].update(reward):  # if a detection was flagged for a aspecific arm, then we need to take note of that and to re-initialize the list of valid rewards, for that arm
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arms[pulled_arm] = []
                self.change_detection[pulled_arm].reset()
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])  # also we need to compute the right empirical mean, over only the valid samples
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])
        for a in range(self.n_cells):  # and also the confidences have to be updated only on the basis of the valid samples (of rewards), for each arm
            n_samples = len(self.valid_rewards_per_arms[a])
            self.confidence[a] = (2*np.log(total_valid_samples)/n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.reward_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)  # notice that here we're also updating the VALID rewards for each arm
        self.collected_rewards = np.append(self.collected_rewards, reward)  # we update the rewards cumulatively
