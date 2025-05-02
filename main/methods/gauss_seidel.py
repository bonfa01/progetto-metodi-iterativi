import numpy as np
from .base import IterativeSolver
import time

class GaussSeidelSolver(IterativeSolver):
    def solve(self):
        x = self.x0.copy()
        start_time = time.time()

        for k in range(self.max_iter):
            x_new = np.copy(x)
            for i in range(self.n):
                sigma = sum(self.A[i][j] * x_new[j] for j in range(self.n) if j != i)
                x_new[i] = (self.b[i] - sigma) / self.A[i][i]
            if np.linalg.norm(self.A @ x_new - self.b) / np.linalg.norm(self.b) < self.tol:
                end_time = time.time()
                return x_new, k + 1, end_time - start_time
            x = x_new

        return x, self.max_iter, time.time() - start_time
