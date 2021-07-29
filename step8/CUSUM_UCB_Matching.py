import numpy as np
from numpy.core.fromnumeric import shape
import config_8
from CUSUM import CUSUM
from scipy.optimize import linear_sum_assignment

np.random.seed(1234)

class CUSUM_UCB_Matching():
    def __init__(self, n_rows, n_cols, params, alpha=0.01):
        self.t = 0

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_cells = self.n_rows * self.n_cols # num of cells in the matching adjacency matrix

        self.empirical_means = np.zeros(self.n_cells)
        # Nt in the computation of the confidences is the number of times that I've matched, at the current time step,
        # a customer of a specific class with a specific promo, starting from the last change detection
        self.confidences = np.array([np.inf] * self.n_cells)

        # here I have a CUSUM for each of the 4 x 4 = 16 cells in the CUSUM_MATCHING matrix
        self.change_detection = [CUSUM(*params) for _ in range(self.n_cells)]
        self.valid_rewards_per_cell = [[] for _ in range(self.n_cells)]
        self.detections = [[] for _ in range(self.n_cells)]  # to keep track of how many times a warning has been raised for a specific cell
        self.alpha = alpha

    
    def pull_cells(self, n_promos, expected_customers):

        def build_matrix(upper_confidence, n_promos, expected_customers):
            # repeat columns
            matrix = np.repeat(upper_confidence, n_promos, axis=1)
            # repeat rows
            matrix = np.repeat(matrix, expected_customers, axis=0)
            return matrix

        def compute_matching(upper_confidence, n_promos, expected_customers):
            m = build_matrix(upper_confidence, n_promos, expected_customers)
            rows, cols = linear_sum_assignment(m, maximize=True)
            matching_mask = np.zeros(m.shape, dtype=int)
            matching_mask[rows, cols] = 1
            return matching_mask * m, matching_mask

        if np.random.binomial(1, 1 - self.alpha):  # we are going to execute this block of code with probability 1-alpha (exploitation)
            upper_confidence = (self.empirical_means + self.confidences).reshape((self.n_rows, self.n_cols))
            upper_confidence[np.isinf(upper_confidence)] = 1e3
            matching, mask = compute_matching(upper_confidence, n_promos, expected_customers)

        else:  # with probability alpha we get a random matching, pulling, in this way, random arms (exploration)
            random_costs = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            # returning --> row_ind, col_ind, matching, matching_mask RANDOM!!
            matching, mask = compute_matching(random_costs)
        
        return matching, mask

    # array = [(customer, promo, reward)]
    # array_2 = [(customer, promo, np.sum(reward, np.where((customer, promo) == (elem[0], elem[1])))) for elem in array]    

    # to be changed (customer, promo, reward) to be passed instead of pulled_arms
    def update(self, c_class, promo, normalized_profit):
        self.t += 1
        flat_cell_index = np.ravel_multi_index((c_class, promo), (self.n_rows, self.n_cols))
        #for pulled_arm, reward in zip(pulled_arm_flat, rewards):
        if self.change_detection[flat_cell_index].update(normalized_profit):  # if a detection was flagged for a aspecific arm, then we need to take note of that and to re-initialize the list of valid rewards, for that arm
            # detections --> time step of the last change detected
            self.detections[flat_cell_index].append(self.t)
            self.valid_rewards_per_cell[flat_cell_index] = []
            self.change_detection[flat_cell_index].reset()
        # self.update_observations(flat_cell_index, normalized_profit)
        self.valid_rewards_per_cell[flat_cell_index].append(normalized_profit)
        self.empirical_means[flat_cell_index] = np.mean(self.valid_rewards_per_cell[flat_cell_index])  # also we need to compute the right empirical mean, over only the valid samples

    def update_confidence(self):
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_cell])
        for cell in range(self.n_cells):  # and also the confidences have to be updated only on the basis of the valid samples (of rewards), for each cell
            n_samples = len(self.valid_rewards_per_cell[cell])
            self.confidences[cell] = (2*np.log(total_valid_samples)/n_samples) ** 0.5 if n_samples > 0 else np.inf

    # def update_observations(self, cell_index, normalized_profit):
    #     self.reward_per_arm[pulled_arm].append(reward)
    #     self.valid_rewards_per_arms[pulled_arm].append(reward)  # notice that here we're also updating the VALID rewards for each arm
    #     self.collected_rewards = np.append(self.collected_rewards, reward)  # we update the rewards cumulatively
