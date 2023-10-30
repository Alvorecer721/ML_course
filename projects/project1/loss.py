import numpy as np


def compute_mse(y, tx, w):
    """Calculate the loss using either MSE"""
    err = y - tx @ w
    mse = np.mean(err**2) / 2
    return mse


def compute_rmse(y, tx, w):
    """Calculate the loss using RMSE"""
    mse = compute_mse(y, tx, w)
    return np.sqrt(2 * mse)


def sigmoid(t):
    """Calculate the sigmoid function"""
    return (1 + np.exp(-t)) ** -1


def compute_negative_log_likelihood(y, tx, w):
    """Calculate the cost by negative log likelihood for logistic regression with label 0 and 1"""
    pred_prob = sigmoid(tx @ w)
    loss = -np.sum(y * np.log(pred_prob) + (1 - y) * np.log(1 - pred_prob)) / y.shape[0]
    return loss


def compute_focal_loss(y, tx, w, alpha=0.25, gamma=2.0):
    """Calculate the focal loss for logistic regression with label 0 and 1"""
    pred_prob = sigmoid(tx @ w)

    epsilon = 1e-8
    pred_prob = np.clip(pred_prob, epsilon, 1.0 - epsilon)

    pos_term = -alpha * y * (1 - pred_prob) ** gamma * np.log(pred_prob)
    neg_term = -(1 - alpha) * (1 - y) * pred_prob**gamma * np.log(1 - pred_prob)

    loss = np.sum(pos_term + neg_term) / y.shape[0]
    return loss
