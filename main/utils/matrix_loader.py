import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

def load_matrix(filepath):
    """
    Carica una matrice in formato Matrix Market (.mtx)
    e la converte in formato sparso CSR (Compressed Sparse Row).
    """
    A = mmread(filepath)
    return csr_matrix(A)

def generate_b_and_x_exact(A):
    """
    Genera il vettore soluzione esatta x = [1, ..., 1]
    e calcola il membro destro b = A * x
    """
    n = A.shape[0]
    x_exact = np.ones(n)
    b = A.dot(x_exact)
    return b, x_exact
