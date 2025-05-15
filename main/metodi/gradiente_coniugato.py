from typing import Optional
from .BaseIterativeSolver import BaseIterativeSolver 
import numpy as np
from  .BaseIterativeSolver import default_stopping_criterion, StoppingCriterion  


class ConjugateGradientSolver(BaseIterativeSolver):
    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:

        self.tol = tol if tol is not None else self.tol
        self.max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        x = np.zeros_like(self.b)
        r = self.b - self.A @ x
        p = r.copy()
        b_norm = np.linalg.norm(self.b)

        while stopping_criterion(r, b_norm, self.tol, self._iterations, self.max_iter):
            Ap = self.A @ p
            alpha = np.dot(r, r) / np.dot(p, Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new
            self._residuals.append(r)
            self._iterations += 1

        self._solution = x
        return x
