import numpy as np
from utils.io import load_matrix, generate_exact_solution, compute_rhs
from methods.jacobi import JacobiSolver
from methods.gauss_seidel import GaussSeidelSolver
from methods.gradient import GradientSolver
from methods.conjugate_gradient import ConjugateGradientSolver
from tabulate import tabulate

def main():
    # === Parametri di input ===
    matrix_paths = [
        "matrices/spa1.mtx", 
        "matrices/spa2.mtx", 
        "matrices/vem1.mtx", 
        "matrices/vem2.mtx"
    ]
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iter = 20000

    # === Ciclo su tutte le matrici ===
    for matrix_path in matrix_paths:
        print(f"\nCaricamento matrice: {matrix_paths}")

        # === Carica matrice A e costruisci x, b ===
        A = load_matrix(matrix_path)
        x_exact = generate_exact_solution(A.shape[0])
        b = compute_rhs(A, x_exact)

        methods = [
            ("Jacobi", JacobiSolver),
            ("Gauss-Seidel", GaussSeidelSolver),
            ("Gradient", GradientSolver),
            ("Conjugate Gradient", ConjugateGradientSolver),
        ]

        # === Ciclo su tutte le tolleranze ===
        for tol in tolerances:
            print(f"\nTolleranza: {tol}")
            results = []

            for name, SolverClass in methods:
                solver = SolverClass(A, b, tol=tol, max_iter=max_iter)
                x_approx, iters, exec_time = solver.solve()
                error = solver.relative_error(x_approx, x_exact)
                results.append([name, iters, f"{error:.2e}", f"{exec_time:.4f} s"])

            print(tabulate(results, headers=["Metodo", "Iterazioni", "Errore Relativo", "Tempo"]))

if __name__ == "__main__":
    main()
