import numpy as np


def compute_mse(y, tx, w):
    """Calculate the loss using either MSE"""
    err = y - tx @ w
    mse = np.mean(err ** 2) / 2
    return mse


def compute_rmse(y, tx, weights):
    """Calculate the loss using RMSE"""
    mse = compute_mse(y, tx, weights)
    return np.sqrt(2 * mse)
