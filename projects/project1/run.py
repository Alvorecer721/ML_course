import os
import numpy as np
from utils import *
from helpers import *
from implementations import *
from DataHandler import DataHandler


class Preprocessing:
    def __init__(self, continuous_cols, nominal_ordinal_cols):
        self.data_handler = DataHandler(
            continuous_cols=continuous_cols,
            nominal_ordinal_cols=nominal_ordinal_cols,
        )

    def load_data(self, path):
        return load_csv_data(os.path.join(path))

    def replace_values(self, data, col, target_value, replacement_value):
        self.data_handler.replace_values_inplace(
            data, col, target_value, replacement_value
        )

    def encode_data(self, x_train, x_test):
        """
        Transfrom data:
        1. Fill NaN values in continuous columns
        2. Standardize continuous columns
        3. One-hot encode nominal/ordinal columns
        """
        return self.data_handler.fit_transform(x_train, x_test)


class Model:
    def __init__(self, max_iter, best_params):
        self.max_iter = max_iter
        self.best_params = best_params

    def train(self, y_train, x_train_encoded):
        tx_train, w = self.add_intercept(x_train_encoded)  # x = [1, x]

        optimal_w, _ = reg_focal_logistic_regression(
            y=y_train[:, 1][:, None],
            tx=tx_train,
            gamma=self.best_params["gamma"],
            max_iters=self.max_iter,
            initial_w=w,
            lambda_=self.best_params["lambda_"],
            alpha=self.best_params["alpha"],
            focal_gamma=self.best_params["focal_gamma"],
            verbose=True,
        )

        return optimal_w

    def predict(self, x_test_encoded, optimal_w):
        tx_test, _ = self.add_intercept(x_test_encoded)
        return pred(
            tx_test,
            optimal_w,
            t=self.best_params["threshold"],
            logistic=True,
            verbose=True,
        )

    def add_intercept(self, X):
        """Add intercept to the data."""

        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.zeros((X.shape[1], 1))
        return X, w


def main():
    """
    Main execution function for training, prediction, and submission generation.

    This function performs the following operations:
    1. Loads the training and test datasets.
    2. Preprocesses the data, including:
       - Feature selection
       - Replacing specific values
       - Encoding nominal and ordinal features
    3. Trains a model using focal logistic regression.
    4. Validates the trained weights against saved weights.
    5. Makes predictions on the test set.
    6. Saves predictions to a submission file.
    """

    selected_features_columns = [
        265,
        0,
        60,
        32,
        52,
        58,
        51,
        26,
        259,
        48,
        38,
        233,
        253,
        39,
        284,
        278,
        279,
        232,
        30,
        28,
        27,
        69,
        50,
        246,
        44,
        234,
        157,
        47,
        103,
        35,
        287,
    ]

    continuous_indices = [12, 19, 20, 30]

    nominal_ordinal_features_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
    ]

    # Load and preprocess data
    prep = Preprocessing(continuous_indices, nominal_ordinal_features_indices)
    x_train, x_test, y_train, train_ids, test_ids = prep.load_data("data/dataset")
    y_train_backup = y_train.copy()

    # Select specified features from training and test datasets.
    x_train_sel = x_train[:, selected_features_columns]
    x_test_sel = x_test[:, selected_features_columns]

    # Value replacements
    replacements = [
        (19, 88, 0),
        (19, 77, np.nan),
        (19, 99, np.nan),
        (20, 88, 0),
        (20, 77, np.nan),
        (20, 99, np.nan),
        (30, 999, np.nan),
    ]

    # Apply the replacements to training and test datasets.
    for col, target, replacement in replacements:
        prep.replace_values(x_train_sel, col, target, replacement)
        prep.replace_values(x_test_sel, col, target, replacement)

    # Encode the training and test datasets.
    x_train_encoded, x_test_encoded = prep.encode_data(x_train_sel, x_test_sel)

    # Define the maximum iterations and best parameters for the model.
    max_iter = 12000
    best_params = {
        "gamma": 0.9778126960938858,
        "lambda_": 2.174295588381516e-06,
        "alpha": 0.531897178573856,
        "focal_gamma": 3.555231848954843,
        "threshold": 0.43043043043043044,
    }

    # Train the model and obtain optimal weights.
    model = Model(max_iter, best_params)
    optimal_w = model.train(y_train_backup, x_train_encoded)

    # Load saved weights and assert
    loaded_weights = np.load("optimal_weight.npy")

    # Load saved weights and validate against trained weights.
    assert np.allclose(
        optimal_w, loaded_weights, atol=1e-8
    ), "The arrays are not close enough!"

    # Make predictions using the test dataset.
    pred_test = model.predict(x_test_encoded, optimal_w)

    # Save predictions
    create_csv_submission(
        ids=np.array(test_ids),
        y_pred=pred_test,
        name="submission.csv",
    )


if __name__ == "__main__":
    main()
