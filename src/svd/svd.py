import numpy as np

def svd(A):
    # Matrix A
    A = A.astype(float)
    m, n = A.shape

    # Creating Matrix B based on Matrix A 
    B = A.T @ A

    eigenvalues, V = np.linalg.eigh(B)

    # DESC sort
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    # Roots
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))

    #Tolerance 
        #Numerical tolerance used to treat very small singular values as zero to ensure numerical stability

    tol = 1e-10 * singular_values[0]

    # Computing Matrix U
    U = np.zeros((m, n))
    for i in range(n):
        if singular_values[i] > tol:
            U[:, i] = (A @ V[:, i]) / singular_values[i]

    Vt = V.T

    return U, singular_values, Vt

