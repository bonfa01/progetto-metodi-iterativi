import numpy as np
from typing import Optional, Callable

StoppingCriterion = Callable[[np.ndarray, float, float, int, int], bool]

def default_stopping_criterion(r, b_norm, tol, k, max_iter):
    return np.linalg.norm(r) / b_norm > tol and k < max_iter

class BaseIterativeSolver:
    def __init__(self, A: np.ndarray, b: np.ndarray, tol=1e-8, max_iter=20000):
        self.A = A
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self._solution = None
        self._iterations = 0
        self._residuals = []

    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_solution(self) -> np.ndarray:
        return self._solution

    def get_iterations(self) -> int:
        return self._iterations

    def get_residuals(self) -> list:
        return self._residuals
