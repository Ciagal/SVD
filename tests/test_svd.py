import numpy as np
from sklearn.utils.extmath import randomized_svd

from svd.svd import svd           
from svd.synthetic import generate_matrix


def test_shapes_make_sense():
    A = np.random.randn(5, 3)
    U, S, Vt = svd(A)
    assert U.shape == (5, 3)
    assert S.shape == (3,)
    assert Vt.shape == (3, 3)


def test_reconstruction_is_good_full_rank():
    np.random.seed(0)
    A = np.random.randn(6, 4)
    U, S, Vt = svd(A)
    A_rec = U @ np.diag(S) @ Vt
    err = np.linalg.norm(A - A_rec, ord="fro")
    assert err < 1e-8


def test_singular_values_nonnegative_sorted():
    A = np.random.randn(7, 5)
    _, S, _ = svd(A)
    assert np.all(S >= -1e-12)
    assert np.all(S[:-1] >= S[1:] - 1e-12)


def test_orthonormality():
    A = np.random.randn(8, 5)
    U, S, Vt = svd(A)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-6)
    assert np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=1e-6)


def test_compare_with_sklearn_topk_reconstruction():
    np.random.seed(1)
    A = np.random.randn(80, 60)

    U, S, Vt = svd(A)
    k = 5
    A_my = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    Us, Ss, Vts = randomized_svd(A, n_components=k, random_state=0, flip_sign=True)
    A_skl = Us @ np.diag(Ss) @ Vts

    err_my = np.linalg.norm(A - A_my, ord="fro")
    err_skl = np.linalg.norm(A - A_skl, ord="fro")

    assert abs(err_my - err_skl) / err_my < 0.02


def test_synthetic_denoising_rank_r_is_better_than_noisy_fit():
    A_clean, A_noisy = generate_matrix()
    U, S, Vt = svd(A_noisy)

    r = 5  
    A_hat = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]

    err_noisy = np.linalg.norm(A_noisy - A_hat, ord="fro")
    err_clean = np.linalg.norm(A_clean - A_hat, ord="fro")

    assert err_clean < err_noisy
