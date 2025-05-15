from typing import Optional
from .BaseIterativeSolver import BaseIterativeSolver 
import numpy as np
from  .BaseIterativeSolver import default_stopping_criterion, StoppingCriterion  

class GaussSeidelSolver(BaseIterativeSolver):
    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:

        self.tol = tol if tol is not None else self.tol
        self.max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]
        x = np.zeros(n)
        r = self.b - self.A @ x
        b_norm = np.linalg.norm(self.b)

        while stopping_criterion(r, b_norm, self.tol, self._iterations, self.max_iter):
            for i in range(n):
                sigma = np.dot(self.A[i, :i], x[:i]) + np.dot(self.A[i, i+1:], x[i+1:])
                x[i] = (self.b[i] - sigma) / self.A[i, i]
            r = self.b - self.A @ x
            self._residuals.append(r)
            self._iterations += 1

        self._solution = x
        return x
