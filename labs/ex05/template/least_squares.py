# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    A = tx.T @ tx
    b = tx.T @ y
    optimal_weight = np.linalg.solve(A, b)

    e = y - tx.dot(optimal_weight)
    mse = (e.T.dot(e)) / (2 * len(y))
    return optimal_weight, mse
