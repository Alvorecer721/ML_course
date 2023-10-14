import numpy as np


def calc_metrics(y_true, y_pred):
    """compute the accuracy, and weighted f1-score."""
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)

    # List of unique classes in the true labels
    classes = np.unique(y_true)

    # Initialize scores
    f1_scores = []
    supports = []

    for c in classes:
        # For each class, calculate precision, recall, and F1

        # True positives
        tp = np.sum((y_pred == c) & (y_true == c))

        # False positives
        fp = np.sum((y_pred == c) & (y_true != c))

        # False negatives
        fn = np.sum((y_pred != c) & (y_true == c))

        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        # F1-score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        f1_scores.append(f1)
        supports.append(np.sum(y_true == c))

    # Calculate weighted F1
    weighted_f1 = np.sum(np.array(f1_scores) * np.array(supports)) / np.sum(supports)

    return accuracy, weighted_f1
