import time
import numpy as np
import os
from scipy.io import mmread
from test import analisi

# Importa i solver dal pacchetto "metodi"
from metodi.jacobi import JacobiSolver
from metodi.gauss_seidel import GaussSeidelSolver
from metodi.gradiente import GradientSolver
from metodi.gradiente_coniugato import ConjugateGradientSolver

def run_solver(name, solver_class, A, b, x_exact, tol):
    solver = solver_class(A, b, tol=tol)
    start_time = time.time()
    x_approx = solver.solve()
    elapsed_time = time.time() - start_time
    relative_error = np.linalg.norm(x_approx - x_exact) / np.linalg.norm(x_exact)
    iterations = solver.get_iterations()

    print(f"--- {name} ---")
    print(f"Errore relativo: {relative_error:.2e}")
    print(f"Iterazioni: {iterations}")
    print(f"Tempo: {elapsed_time:.4f} s\n")

    # Ritorna i dati per l’analisi
    return {
        'Metodo': name,
        'Tolleranza': tol,
        'Iterazioni': iterations,
        'Tempo': elapsed_time,
        'Errore': relative_error
    }


def main():
    file_paths = [
        "/Users/davidebonfanti/Desktop/progetto-metodi-iterativi/main/matrici/spa1.mtx",
        "/Users/davidebonfanti/Desktop/progetto-metodi-iterativi/main/matrici/spa2.mtx",
        "/Users/davidebonfanti/Desktop/progetto-metodi-iterativi/main/matrici/vem1.mtx",
        "/Users/davidebonfanti/Desktop/progetto-metodi-iterativi/main/matrici/vem2.mtx"
    ]
    
    all_results = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Il file {file_path} non esiste.")
        
        A = mmread(file_path).tocsc()
        A = A.toarray()

        n = A.shape[0]
        x_exact = np.ones(n)
        b = A @ x_exact

        matrice_name = os.path.basename(file_path)

        print(f"\n--- File: {matrice_name} ---")
        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
        for tol in tolerances:
            print(f"===== Tolleranza: {tol:.0e} =====")
            for solver_name, solver_class in [
                ("Jacobi", JacobiSolver),
                ("Gauss-Seidel", GaussSeidelSolver),
                ("Gradiente", GradientSolver),
                ("Gradiente Coniugato", ConjugateGradientSolver)
            ]:
                result = run_solver(solver_name, solver_class, A, b, x_exact, tol)
                result['Matrice'] = matrice_name
                all_results.append(result)

    return all_results


if __name__ == "__main__":
    risultati = main()  # esegue i test e cattura i dati
    analisi.dati(risultati)


