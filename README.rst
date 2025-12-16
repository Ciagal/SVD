.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/SVD.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/SVD
    .. image:: https://readthedocs.org/projects/SVD/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://SVD.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/SVD/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/SVD
    .. image:: https://img.shields.io/pypi/v/SVD.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/SVD/
    .. image:: https://img.shields.io/conda/vn/conda-forge/SVD.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/SVD
    .. image:: https://pepy.tech/badge/SVD/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/SVD
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/SVD

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===
SVD
===


    Add a short description here!

# Singular Value Decomposition (SVD) – implementation and comparison

## 📌 Project overview

This project implements **Singular Value Decomposition (SVD)** from scratch using **NumPy** and compares it with the implementation available in **scikit-learn**.  
The main focus is on **low-rank matrix approximation** and **denoising** of synthetic data.

The project demonstrates that truncated SVD:
- captures the dominant structure of data,
- removes noise,
- produces results comparable to `sklearn.utils.extmath.randomized_svd`.

---

## 🎯 Goal of the project

The goal of this project is to:
- implement SVD manually using linear algebra concepts,
- apply it to low-rank matrix approximation,
- analyze reconstruction errors for noisy and clean data,
- compare results with the scikit-learn implementation.

---

## ❓ Problem statement

Given a noisy data matrix \( A \), how well can truncated SVD:
- approximate the original matrix,
- recover the underlying low-rank structure,
- and reduce noise compared to a full noisy reconstruction?

---

## 🧠 Method

### Implemented SVD

The SVD is computed as follows:

1. Compute:
   \[
   B = A^T A
   \]
2. Perform eigenvalue decomposition:
   \[
   B = V \Lambda V^T
   \]
3. Singular values:
   \[
   \sigma_i = \sqrt{\lambda_i}
   \]
4. Left singular vectors:
   \[
   U_i = \frac{A v_i}{\sigma_i}
   \]

This approach follows the mathematical definition of SVD:
\[
A = U \Sigma V^T
\]

---

### Synthetic dataset

A synthetic **low-rank matrix** is generated as:

\[
A_{clean} = U S V^T
\]

where:
- rank \( r = 5 \),
- Gaussian noise is added to obtain a noisy matrix:
\[
A_{noisy} = A_{clean} + \text{noise}
\]

---

## 📊 Experiments and results

### Reconstruction error

Reconstruction error is measured using the Frobenius norm:

\[
\| A - \hat{A} \|_F
\]

Experiments are performed for different ranks \( k \in \{1,2,3,5,10,20\} \).

Key observations:
- For noisy data, the reconstruction error decreases with increasing \( k \).
- For clean data, the lowest error is achieved around the true rank \( r = 5 \).
- Using larger \( k \) for clean data leads to fitting noise and slightly higher error.

---

### Comparison with scikit-learn

Results show that:
- Singular values closely match those from `randomized_svd`.
- Reconstruction errors differ only by numerical precision.
- The custom implementation behaves consistently with scikit-learn.

---

## 🧪 Tests

The project includes automated tests using **pytest**, covering:
- output shapes of SVD,
- reconstruction accuracy,
- non-negativity and ordering of singular values,
- orthonormality of matrices \( U \) and \( V \),
- comparison with scikit-learn,
- denoising effectiveness for synthetic data.

---

## 📂 Project structure




.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
