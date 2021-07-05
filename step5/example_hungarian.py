import config_5
from Environment_5 import *
import numpy as np
from tqdm import tqdm
from hungarian_algorithm import *
from Learner_5 import *


np.random.seed(1234)

for e in tqdm(config_5.N_EXPS):
    env = Environment_5(n_arms=config_5.N_ARMS, cr1=config_5.CR1, cr2=config_5.CR2)
    h_learner = Learner_5()

    for t in range(config_5.T):

        # hungarian algorithm
        optimal_matching = h_learner.compute_matching()
