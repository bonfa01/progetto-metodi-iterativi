import numpy as np
from .base import IterativeSolver
import time

class ConjugateGradientSolver(IterativeSolver):
    def solve(self):
        x = self.x0.copy()
        r = self.b - self.A @ x
        p = r.copy()
        rs_old = np.dot(r, r)
        start_time = time.time()

        for k in range(self.max_iter):
            Ap = self.A @ p
            alpha = rs_old / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.dot(r, r)
            if np.sqrt(rs_new) / np.linalg.norm(self.b) < self.tol:
                end_time = time.time()
                return x, k + 1, end_time - start_time
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x, self.max_iter, time.time() - start_time
