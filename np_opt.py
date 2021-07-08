# weighted_averages step3, step4 -------------------------------------------------------------------------------
def fun(i, arm):
    cr1 = np.random.beta(arm[:, 0], arm[:, 1]) # 4x1
    margins1 = config_4.MARGINS_1[i] # 1x1
    cr2 = np.random.beta(self.beta_cr2[:, :, 0], self.beta_cr2[:, :, 1]) # 4x4
    matching_prob = config_4.MATCHING_PROB * np.sum(self.expected_customers) / np.expand_dims(self.expected_customers, axis=0).T # 4x4
    margins2 = config_4.MARGINS_2 # 4x1

    a = cr1 * (margins1 + np.dot(cr2 * matching_prob, margins2)) #   4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                 # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                 # = 4x1 * (1x1 + 4x1) = 
                                                                 # = 4x1 * 4x1 = 
                                                                 # = 4x1

    #return np.average(a, weights=self.expected_customers)
    return np.dot(a, self.expected_customers)

weighted_averages = [fun(i, arm) for i, arm in enumerate(self.beta_parameters)]


# n_promos step5 -----------------------------------------------------------------------------------------------
n_promos = (config_5.PROMO_PROB[1 :] * matrix_dim).astype(int)
n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))


# customers' rows step5 ----------------------------------------------------------------------------------------
matrix = np.array([])
for i, n in enumerate(n_promos):
    matrix = np.append(matrix, [config_5.MARGIN_1 * np.random.beta(self.beta_cr1[:, 0], self.beta_cr1[:, 1]) + config_5.MARGINS_2 * np.random.beta(self.beta_cr2[i, :, 0], self.beta_cr2[i, :, 0]) for _ in range(n)])
matrix = -1 * matrix

######### matrix = np.array([customer0_row, customer1_row, customer2_row, customer3_row])

# matrix building step5 ----------------------------------------------------------------------------------------
matrix = np.repeat(matrix, self.expected_customers, axis=0)

