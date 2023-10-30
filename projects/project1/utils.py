from loss import *


def compute_gradient(y, tx, w, logistic=False, focal=False, alpha=0.25, gamma=2.0):
    """
        Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        logistic: boolean, if True, uses logistic regression gradient
        focal: boolean, if True, uses focal loss gradient
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    if logistic:
        prob = sigmoid(tx @ w)
        grad = tx.T @ (prob - y) / y.shape[0]
    elif focal:
        prob = sigmoid(tx @ w)
        modulating_factor = (1 - prob) ** gamma
        focal_weight = (
            alpha * y * modulating_factor + (1 - alpha) * (1 - y) * prob**gamma
        )

        grad = tx.T @ (focal_weight * (prob - y)) / y.shape[0]
    else:
        e = y - tx @ w
        grad = -tx.T @ e / tx.shape[0]

    return grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(features, degree, cols):
    """
        Build polynomial features for specified columns up to a given degree.

    Args:
        features: Input matrix (n_samples, n_features)
        degree: The max polynomial degree
        cols: List of column indices to which polynomial features should be applied

    Returns:
        Extended feature matrix with polynomial features for specified columns
    """

    # Start with the original features
    output_features = [features]

    # For each degree and column index
    for d in range(2, degree + 1):
        for col in cols:
            # Compute the polynomial feature for this degree and column
            poly_feature = features[:, col] ** d
            output_features.append(poly_feature.reshape(-1, 1))

    # Combine all the features
    return np.hstack(output_features)


from itertools import combinations


def build_interaction(data, columns=None):
    """
    Add interaction terms to the data array.

    Parameters:
    - data: numpy array
        The original data array.
    - columns: list of int
        A list containing indices representing the columns to be interacted.
        If None, all columns are used.

    Returns:
    - A new numpy array with interaction terms added as new columns.
    """
    if columns is None:
        columns = range(data.shape[1])

    # Generating all possible pairs of columns to interact
    interaction_pairs = combinations(columns, 2)

    # Making a copy of the data to not modify the original array
    new_data = np.copy(data)

    for col1_index, col2_index in interaction_pairs:
        # Calculating the interaction term
        interaction_term = data[:, col1_index] * data[:, col2_index]

        # Reshaping the interaction term to have the same number of dimensions as the data
        interaction_term = interaction_term.reshape(-1, 1)

        # Adding the interaction term as a new column
        new_data = np.hstack((new_data, interaction_term))

    return new_data


def build_k_indices(y, k_fold, seed):
    """
        Build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(model, loss_fn, y, x, k_indices, k, *args, **kwargs):
    """
        Return the loss for a given model and a fold corresponding to k_indices.

    Args:
        model:      callable, the model to be used
        loss_fn:    callable, the loss function to be used
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold
        *args:      positional arguments to pass to the model
        **kwargs:   keyword arguments to pass to the model

    Returns:
        train and test loss (MSE or RMSE depending on your choice)
    """

    # get k'th subgroup in test, others in train
    valid_indices = k_indices[k]
    train_indices = k_indices[np.delete(range(len(k_indices)), k, axis=0)].flatten()

    train_x = x[train_indices]
    valid_x = x[valid_indices]

    # Train using the provided model function
    w, _ = model(y[train_indices], train_x, *args, **kwargs)

    # Calculate the loss for train and test data
    loss_tr = loss_fn(y[train_indices], train_x, w)
    loss_te = loss_fn(y[valid_indices], valid_x, w)

    return loss_tr, loss_te


def cross_validation_over_gamma(model, loss_fn, y, tx, k_indices, max_iters, gammas):
    """
    This function is exclusively for mean square error model and logistic regression model.
    The cross validation is performed over a range of gamma values.

    One iteration of mean_squared_error_sgd takes 3s
    One iteration of mean_squared_error_gd takes 6.4s
    """
    best_loss = float("inf")
    best_gamma = None
    loss_results = []  # to store average losses for each gamma

    tx, _ = add_intercept(tx)

    # Iterate through all gamma values
    for gamma in gammas:
        total_loss = 0

        # k-fold cross-validation for each gamma
        for k in range(len(k_indices)):
            # each fold, we re-initialise weights to 0
            initial_w = np.zeros((tx.shape[1], 1))

            _, loss_te = cross_validation(
                model,
                loss_fn,
                y,
                tx,
                k_indices,
                k,
                initial_w=initial_w,
                max_iters=max_iters,
                gamma=gamma,
            )
            total_loss += loss_te

        average_loss = total_loss / len(k_indices)
        loss_results.append(average_loss)

        # Update the best gamma if necessary
        if average_loss < best_loss:
            best_loss = average_loss
            best_gamma = gamma

    return best_gamma, best_loss


def pred(tx, w, t=0, logistic=False, verbose=True):
    """
        Compute the prediction of labels given features and weights.

    Args:
        tx:  Feature matrix of shape (N, D)
        w:    Weight vector of shape (D, 1)
        t: Threshold for distinguishing 1 from -1. Defaults to 0.
        logistic:  True if using a (regularised) logistic regression . Defaults to False.
        verbose:   Print the range of the prediction. Defaults to True.

    Returns:
        labels: predicted labels of shape (N, 1)
    """

    if logistic:
        pred = sigmoid(tx @ w)
    else:
        pred = tx @ w

    if verbose is True:
        # Get the range
        min_pred, max_pred = np.min(pred), np.max(pred)
        print(f"Range of prediction: [{min_pred}, {max_pred}]")

    labels = np.where(pred >= t, 1, -1)
    return labels


def calc_metrics(y_true, y_pred):
    """
        Compute the accuracy and F1-score for binary classification with labels 1 and -1.
        Keep the shape if y_true consistent with y_pred to fasten the computation.

    Args:
        y_true: true labels of shape (N, 1)
        y_pred: predicted labels of shape (N, 1)
    """
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)

    # Assuming the positive class is labeled as 1 (and the negative class as -1)

    # True positives
    tp = np.sum((y_pred == 1) & (y_true == 1))

    # False positives
    fp = np.sum((y_pred == 1) & (y_true == -1))

    # False negatives
    fn = np.sum((y_pred == -1) & (y_true == 1))

    # Precision and recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # F1-score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return accuracy, f1
