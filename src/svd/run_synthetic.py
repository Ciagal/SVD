import numpy as np
from sklearn.utils.extmath import randomized_svd

from svd import svd
from synthetic import generate_matrix


A_clean, A = generate_matrix()


U, S, Vt = svd(A)

k = 5
U_k = U[:, :k]
S_k = S[:k]
Vt_k = Vt[:k, :]

A_my = U_k @ np.diag(S_k) @ Vt_k

U_s, S_s, Vt_s = randomized_svd(A, n_components=k, random_state=0)
A_sklearn = U_s @ np.diag(S_s) @ Vt_s

# Errors
err_my = np.linalg.norm(A - A_my)
err_skl = np.linalg.norm(A - A_sklearn)

err_my_clean = np.linalg.norm(A_clean - A_my)
err_skl_clean = np.linalg.norm(A_clean - A_sklearn)

print("MY SVD error (noisy):     ", err_my)
print("SKLEARN error (noisy):    ", err_skl)
print("MY SVD error (clean):     ", err_my_clean)
print("SKLEARN error (clean):    ", err_skl_clean)

print("\nTop singular values (my):", S_k)
print("Top singular values (sklearn):", S_s)
