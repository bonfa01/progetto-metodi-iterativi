import numpy as np
from .base import IterativeSolver
import time

class GradientSolver(IterativeSolver):
    def solve(self):
        x = self.x0.copy()
        r = self.b - self.A @ x
        start_time = time.time()

        for k in range(self.max_iter):
            Ar = self.A @ r
            alpha = np.dot(r, r) / np.dot(r, Ar)
            x = x + alpha * r
            r = self.b - self.A @ x
            if np.linalg.norm(r) / np.linalg.norm(self.b) < self.tol:
                end_time = time.time()
                return x, k + 1, end_time - start_time

        return x, self.max_iter, time.time() - start_time
