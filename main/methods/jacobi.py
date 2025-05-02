import numpy as np
import time
from .base import IterativeSolver

class JacobiSolver(IterativeSolver):
    def solve(self):
        x = self.x0.copy()
        D = np.diag(self.A)
        R = self.A - np.diagflat(D)
        start_time = time.time()
        
        for k in range(self.max_iter):
            x_new = (self.b - R @ x) / D
            if np.linalg.norm(self.A @ x_new - self.b) / np.linalg.norm(self.b) < self.tol:
                end_time = time.time()
                return x_new, k + 1, end_time - start_time
            x = x_new
        
        return x, self.max_iter, time.time() - start_time
