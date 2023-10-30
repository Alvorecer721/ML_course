from utils import (
    compute_mse,
    compute_gradient,
    compute_negative_log_likelihood,
    compute_focal_loss,
    batch_iter,
)
import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
        Linear regression using gradient descent

    Args:
        y:         numpy array of shape=(N, )
        tx:        numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma:     a scalar denoting the stepsize

    Returns:
        loss
        w
    """
    w = initial_w

    for _ in range(max_iters):
        # compute gradient
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    # compute loss
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
        Linear regression using stochastic gradient descent

    Args:
        y:         numpy array of shape=(N,1)
        tx:        numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a   scalar denoting the stepsize

    Returns:
        w:    the same shape as initial_w
        loss: scalar
    """

    w = initial_w

    for _ in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1):
            # compute gradient and loss
            grad = compute_gradient(batch_y, batch_tx, w)

            # update w
            w = w - gamma * grad

    # compute loss
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """
        Calculate the least squares solution.
        returns mse, and optimal weights.

    Args:
        y:  numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w:    optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
        Implement ridge regression.

    Args:
        y:       numpy array of shape (N,), N is the number of samples.
        tx:      numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w:    Optimised weights, numpy array of shape(D, 1)
        loss: Final loss value, scalar.
    """
    A = tx.T @ tx + lambda_ * 2 * len(y) * np.identity(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)  # Loss without the penalty term
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
        This gradient descent-based logistic regression implementation is designed mainly for labels of 0 and 1.
        Hence, we initially verify the labels and adjust them to 0 or 1 as needed.

    Args:
        y:         A numpy array of shape (N, 1) containing binary labels (0 or 1).
        tx:        A numpy array of shape (N, D) containing the training data.
        initial_w: Initial weights of shape (D, 1).
        max_iters: Maximum number of iterations for the gradient descent.
        gamma:     Learning rate for the gradient descent.

    Returns:
        tuple: Returns the optimised weights and the final loss value.
    """
    # Replace -1 with 0 in y
    y[y == -1] = 0

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w, logistic=True)
        w -= gamma * grad

        # log info
        if n_iter % 100 == 0:
            loss = compute_negative_log_likelihood(y, tx, w)
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))

    loss = compute_negative_log_likelihood(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, verbose=False):
    """
        This gradient descent-based regularised logistic regression implementation is designed mainly for labels of 0 and 1.
        Hence, we initially verify the labels and adjust them to 0 or 1 as needed.

    Args:
        y:         A numpy array of shape (N, 1) containing binary labels (0 or 1).
        tx:        A numpy array of shape (N, D) containing the training data.
        lambda_:   Regularization parameter.
        initial_w: Initial weights of shape (D, 1).
        max_iters: Maximum number of iterations for the gradient descent.
        gamma:     Learning rate for the gradient descent.

    Returns:
        w:    Optimised weights, numpy array of shape(D, 1)
        loss: Final loss value, scalar.
    """
    # Replace -1 with 0 in y
    y[y == -1] = 0
    assert np.all(np.unique(y) == np.array([0, 1])), "y must contain only 0 or 1."

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w, logistic=True) + 2 * lambda_ * w
        w -= gamma * grad

        if verbose and n_iter % 100 == 0:
            loss = compute_negative_log_likelihood(y, tx, w)
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))

    loss = compute_negative_log_likelihood(y, tx, w)
    return w, loss


def reg_focal_logistic_regression(
    y,
    tx,
    initial_w,
    max_iters,
    gamma,
    lambda_=0.0,
    alpha=0.25,
    focal_gamma=3.0,
    verbose=False,
):
    """
        Gradient descent-based logistic regression with focal loss designed for labels of 0 and 1.
        We initially verify the labels and adjust them to 0 or 1 as needed.

    Args:
        y:           A numpy array of shape (N, 1) containing binary labels (0 or 1).
        tx:          A numpy array of shape (N, D) containing the training data.
        initial_w:   Initial weights of shape (D, 1).
        max_iters:   Maximum number of iterations for the gradient descent.
        gamma:       Learning rate for the gradient descent.
        alpha:       Balancing factor for focal loss.
        focal_gamma: Focusing parameter for focal loss.
        lambda_:     Regularization strength

    Returns:
        tuple: Returns the optimised weights and the final loss value using focal loss.
    """
    # Replace -1 with 0 in y
    y[y == -1] = 0
    assert np.all(np.unique(y) == np.array([0, 1])), "y must contain only 0 or 1."

    w = initial_w

    for n_iter in range(max_iters):
        # Compute gradient based on focal loss
        grad = (
            compute_gradient(y, tx, w, focal=True, alpha=alpha, gamma=focal_gamma)
            + 2 * lambda_ * w
        )
        # Update weights
        w -= gamma * grad

        if verbose and n_iter % 100 == 0:
            # Compute focal loss for current iteration
            loss = compute_focal_loss(y, tx, w, alpha, focal_gamma)
            print("Current iteration={i}, focal loss={l}".format(i=n_iter, l=loss))

    # Compute the final focal loss
    loss = compute_focal_loss(y, tx, w, alpha, focal_gamma)
    return w, loss
