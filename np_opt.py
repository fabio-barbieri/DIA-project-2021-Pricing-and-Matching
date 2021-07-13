# weighted_averages step3, step4 ------------------------------------------------------------------------------- OK
def fun(i, arm):
    cr1 = np.random.beta(arm[:, 0], arm[:, 1]) # 4x1
    margins1 = config_4.MARGINS_1[i] # 1x1
    cr2 = np.random.beta(self.beta_cr2[:, :, 0], self.beta_cr2[:, :, 1]) # 4x4
    matching_prob = config_4.MATCHING_PROB * np.sum(self.expected_customers) / np.expand_dims(self.expected_customers, axis=1) # 4x4
    margins2 = config_4.MARGINS_2 # 4x1

    a = cr1 * (margins1 + np.dot(cr2 * matching_prob, margins2)) #   4x1 * (1x1 + dot(4x4 * 4x4 + 4x1)) = 
                                                                 # = 4x1 * (1x1 + dot(4x4, 4x1) = 
                                                                 # = 4x1 * (1x1 + 4x1) = 
                                                                 # = 4x1 * 4x1 = 
                                                                 # = 4x1

    #return np.average(a, weights=self.expected_customers)
    return np.dot(a, self.expected_customers)

weighted_averages = [fun(i, arm) for i, arm in enumerate(self.beta_parameters)]


# n_promos step5 ----------------------------------------------------------------------------------------------- OK
n_promos = (config_5.PROMO_PROB[1 :] * matrix_dim).astype(int)
n_promos = np.insert(n_promos, 0, matrix_dim - np.sum(n_promos))


# customers' rows step5 ---------------------------------------------------------------------------------------- ASPETTA DI SAPERE VALORE CELLE
matrix = np.zeros((4,0))
profit = np.array(np.arange(16)).reshape((4, 4))
n_promos = [9, 4, 4, 2]
n_customers = [5, 10, 4, 1]
for i, n_promo in enumerate(n_promos):
    matrix = np.column_stack([matrix, np.repeat(profit[:, i].reshape(4, 1),n_promo, axis=1)])
matrix = np.repeat(matrix, n_customers, axis=0)

######### matrix = np.array([customer0_row, customer1_row, customer2_row, customer3_row])

# matrix building step5 ---------------------------------------------------------------------------------------- OK
matrix = np.repeat(matrix, self.expected_customers, axis=0)

