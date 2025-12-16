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

# SVD – custom implementation

This repository contains a custom implementation of **Singular Value Decomposition (SVD)** using NumPy, together with experiments, tests and comparison with scikit-learn.

---

## Overview

The project includes:
- manual implementation of SVD,
- synthetic low-rank data generator,
- experiments on noisy and clean matrices,
- comparison with `sklearn.utils.extmath.randomized_svd`,
- automated tests and coverage.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate

pip install -r requirements.txt

pip install numpy matplotlib scikit-learn pytest tox


##Usage

python src/svd/run_synthetic.py

This script:

-generates synthetic low-rank data,

-computes truncated SVD,

-compares reconstruction errors,

-prints top singular values.

##Run examples with plots

python src/svd/examples.py

This script:

-shows a simple 3 x 3 SVD example,

-visualizes reconstruction errors for different ranks,

-compares results with scikit-learn.

##Tests

pytest

Tests include:

-output shape validation,

-reconstruction accuracy,

-singular value ordering,

-orthonormality of matrices,

-comparison with scikit-learn,

-denoising effectiveness on synthetic data.

##Tox

To run tests in an isolated environment:

pip install tox
tox

To run documentation environment (if configured):

tox -e docs

##Project structure

src/svd
├── svd.py            # custom SVD implementation
├── synthetic.py      # synthetic low-rank data generator
├── run_synthetic.py  # experiment script
├── examples.py       # demos and plots
tests
├── test_svd.py
├── test_skeleton.py
docs
├── Rapport_SVD_Algorithm.pdf



.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
