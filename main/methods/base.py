from abc import ABC, abstractmethod
import numpy as np
import time

class IterativeSolver(ABC):
    def __init__(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-6, max_iter: int = 20000):
        self.A = A
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.n = len(b)
        self.x0 = np.zeros_like(b)

    @abstractmethod
    def solve(self):
        """Implementa il metodo specifico di risoluzione"""
        pass

    def relative_error(self, x_approx, x_exact):
        return np.linalg.norm(x_approx - x_exact) / np.linalg.norm(x_exact)

    def residual_norm(self, x_approx):
        return np.linalg.norm(self.A @ x_approx - self.b) / np.linalg.norm(self.b)
