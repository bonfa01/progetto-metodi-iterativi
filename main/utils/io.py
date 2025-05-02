import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

def load_matrix(file_path: str) -> csr_matrix:
    """
    Legge una matrice da file .mtx e la restituisce in formato CSR.
    """
    A = mmread(file_path).tocsr()
    return A

def generate_exact_solution(n: int) -> np.ndarray:
    """
    Genera il vettore soluzione esatta x = [1, 1, ..., 1]
    """
    return np.ones(n)

def compute_rhs(A, x_exact) -> np.ndarray:
    """
    Calcola b = A @ x esatta
    """
    return A @ x_exact
