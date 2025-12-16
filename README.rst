# SVD – custom implementation

This repository contains a custom implementation of Singular Value Decomposition (SVD) using NumPy, together with experiments, tests and comparison with scikit-learn.

---

## Overview

The project includes:
- manual implementation of SVD
- synthetic low-rank data generator
- experiments on noisy and clean matrices
- comparison with sklearn.utils.extmath.randomized_svd
- automated tests and coverage

---

## Installation

Create and activate a virtual environment:

python -m venv env
source env/bin/activate

Install dependencies:

pip install -r requirements.txt

or manually:

pip install numpy matplotlib scikit-learn pytest tox

---

## Usage

Run synthetic experiment:

python src/svd/run_synthetic.py

This script:
- generates synthetic low-rank data
- computes truncated SVD
- compares reconstruction errors
- prints top singular values

---

## Run examples with plots

python src/svd/examples.py

This script:
- shows a simple 3x3 SVD example
- visualizes reconstruction errors for different ranks
- compares results with scikit-learn

---

## Tests

Run all tests:

pytest

Tests include:
- output shape validation
- reconstruction accuracy
- singular value ordering
- orthonormality of matrices
- comparison with scikit-learn
- denoising effectiveness on synthetic data

---

## Tox

To run tests in an isolated environment:

pip install tox
tox

To run documentation environment (if configured):

tox -e docs

---

## Project structure

src/svd
├── svd.py            custom SVD implementation
├── synthetic.py      synthetic low-rank data generator
├── run_synthetic.py  experiment script
├── examples.py       demos and plots

tests
├── test_svd.py
├── test_skeleton.py

docs
├── Rapport_SVD_Algorithm.pdf

