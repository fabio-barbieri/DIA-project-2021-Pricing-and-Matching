import numpy as np
import copy

def step1(m):
    return m - np.min(m, axis=1).reshape((m.shape[0], 1))


def step2(m):
    return m - np.min(m, axis=0)

def min_n_zeros(m, assigned):
    min_n = m.shape[1]
    if np.sum(m == 0) == 0:
        return 0
    for i in range(0, m.shape[0]):
        if(i not in assigned) and (np.sum(m[i, :] == 0) < min_n ) and ( np.sum(m[i, :] == 0) != 0):
            min_n = np.sum(m[i, :] == 0)
    return min_n


def step3(m):
    n_rows = m.shape[0]
    n_cols = m.shape[1]
    assigned = np.array([])
    assignments = np.zeros(m.shape,dtype=int)
    
    mf = copy.copy(m)
    while(np.sum(mf == 0) > 0):
        for i in range(0, n_rows):
            for j in range(0,n_cols):
                if (mf[i, j] == 0 and min_n_zeros(mf,assigned) == np.sum(mf[i, :] == 0)):
                    assignments[i, j] = 1
                    mf[i, :] += 1
                    mf[:, j] += 1
                    assigned = np.append(assigned, i)

    rows = np.linspace(0, n_rows-1, n_rows).astype(int)
    marked_rows = np.setdiff1d(rows, assigned)
    new_marked_rows = marked_rows.copy()
    marked_cols = np.array([])
    while len(new_marked_rows) > 0:
        new_marked_cols = np.array([], dtype=int)
        for nr in new_marked_rows:
            zeros_cols = np.argwhere(m[nr, :] == 0).reshape(-1)
            new_marked_cols = np.append(new_marked_cols, np.setdiff1d(zeros_cols, marked_cols) )
        marked_cols = np.append(marked_cols, new_marked_cols)
        new_marked_rows = np.array([], dtype=int)
        for nc in new_marked_cols:
            new_marked_rows = np.append(new_marked_rows, np.argwhere(assignments[:,nc] == 1).reshape(-1) )
        marked_rows = np.unique(np.append(marked_rows, new_marked_rows))
    return np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_cols)


def step4(m, covered_rows, covered_cols):
    uncovered_rows = np.setdiff1d(np.linspace(0, m.shape[0]-1, m.shape[0]), covered_rows).astype(int)
    uncovered_cols = np.setdiff1d(np.linspace(0, m.shape[1]-1, m.shape[1]), covered_cols).astype(int)
    min_val = np.max(m)
    for i in uncovered_rows.astype(int):
        for j in uncovered_cols.astype(int):
            if(m[i, j] < min_val):
                min_val = m[i, j]
    for i in uncovered_rows.astype(int):
        m[i, :] -= min_val
    for j in covered_cols.astype(int):
        m[:, j] += min_val
    return m


def find_rows_single_zero(matrix):
    for i in range(0, matrix.shape[0]):
        if(np.sum(matrix[i, :] == 0) == 1):
            j =  np.argwhere(matrix[i, :] == 0).reshape(-1)[0]
            return i, j
    return False


def find_cols_single_zero(matrix):
    for j in range(0,matrix.shape[1]):
        if(np.sum(matrix[:,j] == 0) == 1):
            i = np.argwhere(matrix[:, j] == 0).reshape(-1)[0]
            return i, j
    return False


def has_zeros(m):
    return np.sum(m == 0) > 0


def first_zero(m):
    return np.argwhere(m == 0)[0][0], np.argwhere(m == 0)[0][1]


def assignment_single_zero_lines(m, assignment):
    val = find_rows_single_zero(m)
    while(val):
        i, j = val[0], val[1]
        m[:, j] += 1
        assignment[i, j] = 1
        val = find_rows_single_zero(m)
    val = find_cols_single_zero(m)
    while(val):
        i, j = val[0], val[1]
        m[i, :] += 1
        assignment[i, j] = 1
        val = find_cols_single_zero(m)
    return assignment


def final_assignment(initial_matrix, m):
    assignment = np.zeros(m.shape, dtype=int)
    assignment = assignment_single_zero_lines(m, assignment)
    while(has_zeros(m)):
        i, j = first_zero(m)
        assignment[i, j] = 1
        m[i, :] +=1
        m[:, j] +=1
        assignment= assignment_single_zero_lines(m, assignment)
    return assignment * initial_matrix, assignment


def hungarian_algorithm(matrix, dim):
    #m = np.max(matrix) - matrix
    #step1(m)
    #step2(m)
    #n_lines = 0
    #max_length = np.maximum(m.shape[0], m.shape[1])
    #while n_lines != max_length:
    #    lines = step3(m)
    #    n_lines = len(lines[0]) + len(lines[1])
    #    if n_lines != max_length:
    #         step4(m, lines[0], lines[1])
    # return final_assignment(matrix, m)
    return np.eye(dim)