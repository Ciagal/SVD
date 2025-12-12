import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

from svd import svd
from synthetic import generate_matrix


def simple_matrix():
    print("\nsimple 3x3 matrix ===")
    A = np.array([
        [3, 1, 1],
        [-1, 3, 1],
        [1, 1, 3]
    ], dtype=float)

    U, S, Vt = svd(A)
    A_rec = U @ np.diag(S) @ Vt

    print("A:")
    print(A)
    print("\nSingular values:")
    print(S)
    print("\nA reconstructed:")
    print(A_rec)
    print("\nReconstruction error:", np.linalg.norm(A - A_rec))


def synthetic_low_rank():
    print("\nSynthetic low-rank + noise")
    A_clean, A = generate_matrix()

    # full SVD
    U, S, Vt = svd(A)

    explained = (S**2) / np.sum(S**2)
    print("\nTop singular values:")
    print(S[:10])
    print("\nExplained variance (%):")
    print((explained[:10] * 100))

    ks = [1, 2, 3, 5, 10, 20]
    print("\nk | my_noisy | skl_noisy | my_clean | skl_clean | Sdiff")
    print("-" * 70)

    my_noisy_list = []
    my_clean_list = []
    skl_noisy_list = []
    skl_clean_list = []

    for k in ks:
        # my truncated
        A_my = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        my_noisy = np.linalg.norm(A - A_my)
        my_clean = np.linalg.norm(A_clean - A_my)

        # sklearn truncated
        Us, Ss, Vts = randomized_svd(A, n_components=k, random_state=0, flip_sign=True)
        A_skl = Us @ np.diag(Ss) @ Vts
        skl_noisy = np.linalg.norm(A - A_skl)
        skl_clean = np.linalg.norm(A_clean - A_skl)

        sdiff = np.linalg.norm(S[:k] - Ss)

        print(f"{k:2d} | {my_noisy:7.3f} | {skl_noisy:7.3f} | {my_clean:7.3f} | {skl_clean:7.3f} | {sdiff:.3e}")

        my_noisy_list.append(my_noisy)
        my_clean_list.append(my_clean)
        skl_noisy_list.append(skl_noisy)
        skl_clean_list.append(skl_clean)

    #2 plots (noisy and clean)
    plt.figure()
    plt.plot(ks, my_noisy_list, marker="o", label="my SVD")
    plt.plot(ks, skl_noisy_list, marker="x", linestyle="--", label="sklearn")
    plt.xlabel("k (rank)")
    plt.ylabel("reconstruction error (Fro norm)")
    plt.title("Noisy matrix: error vs k")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(ks, my_clean_list, marker="o", label="my SVD")
    plt.plot(ks, skl_clean_list, marker="x", linestyle="--", label="sklearn")
    plt.xlabel("k (rank)")
    plt.ylabel("reconstruction error (Fro norm)")
    plt.title("Clean matrix: error vs k")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    #3rd plot  - diff: my vs sklearn
    diff_noisy = np.abs(np.array(my_noisy_list) - np.array(skl_noisy_list))
    diff_clean = np.abs(np.array(my_clean_list) - np.array(skl_clean_list))

    plt.figure()
    plt.plot(ks, diff_noisy, marker="o", label="|my - sklearn| (noisy)")
    plt.plot(ks, diff_clean, marker="o", label="|my - sklearn| (clean)")
    plt.xlabel("k (rank)")
    plt.ylabel("absolute error difference")
    plt.title("Difference between my SVD and sklearn")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    simple_matrix()
    synthetic_low_rank()
