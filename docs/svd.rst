SVD Algorithm
=============

The Singular Value Decomposition factorizes a matrix A into three matrices:

A = U Σ Vᵀ

Where:
- U contains left singular vectors
- Σ contains singular values
- V contains right singular vectors

In this project, SVD is computed iteratively using a simple numerical approach.
The correctness is verified by reconstructing the original matrix and measuring
the reconstruction error.