import numpy as np

def generate_matrix():
    np.random.seed(0)

    m = 80
    n = 60
    r = 5

    # Random Matrices
    U = np.random.randn(m, r)
    V = np.random.randn(n, r)

    # Singular Values
    S = np.diag([20, 15, 10, 7, 5])

    A_clean = U @ S @ V.T

    # Adding some noise
    noise = 0.5 * np.random.randn(m, n)
    A_noisy = A_clean + noise

    return A_clean, A_noisy
