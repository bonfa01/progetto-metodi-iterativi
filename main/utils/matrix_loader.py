import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

def load_matrix(filepath):
    A = mmread(filepath)
    return csr_matrix(A)

def generate_b_and_x_exact(A):
    n = A.shape[0]
    x_exact = np.ones(n)
    b = A.dot(x_exact)
    return b, x_exact
