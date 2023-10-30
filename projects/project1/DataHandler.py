import numpy as np


class DataHandler:
    def __init__(self, continuous_cols, nominal_ordinal_cols):
        self.train_mean = None
        self.train_std = None
        self.continuous_cols = continuous_cols
        self.nominal_ordinal_cols = nominal_ordinal_cols

    def fit(self, X):
        """Compute the mean and std deviation of continuous columns for later scaling."""
        self.train_mean = np.nanmean(X[:, self.continuous_cols], axis=0)
        self.train_std = np.nanstd(X[:, self.continuous_cols], axis=0)

    def transform(self, X):
        """
        Transfrom data:
        1. Fill NaN values in continuous columns
        2. Standardize continuous columns
        3. One-hot encode nominal/ordinal columns
        """
        X = self._fill_na(X)  # Fill NaN values first
        X = self._standardize(X)
        X = self._one_hot_encode(X)
        return X

    def fit_transform(self, X_train, X_test):
        """Fit and transform X, X_train and X_test must be given for consistance in one-hot encoding."""
        # Combine X_train and X_test
        X = np.vstack([X_train, X_test])

        # Fit on the training data only to avoid data leakage
        self.fit(X_train)

        # Transform the combined dataset
        transformed_X = self.transform(X)

        # Split the transformed dataset back into training and testing components
        transformed_X_train = transformed_X[: len(X_train)]
        transformed_X_test = transformed_X[len(X_train) :]

        return transformed_X_train, transformed_X_test

    def replace_values_inplace(self, X, col, target_value, replacement_value):
        """
        Replace a specific value inside a specified column with another value, in-place.

        Parameters:
        - X: numpy array
            Input data array.
        - col: int
            Column index to be processed.
        - target_value: int/float
            Value to be replaced.
        - replacement_value: int/float/np.nan
            Value to replace the target_value with.
        """

        X[X[:, col] == target_value, col] = replacement_value

    def stratified_shuffle_split(self, X, y, val_size=0.2, random_state=1):
        """
        Split the data into training and validation sets while preserving
        the proportion of each class in the binary target variable using only numpy.

        Parameters:
        - X: numpy array
            Feature matrix.
        - y: numpy array
            Two-column array where the second column is the target vector (binary: -1 or 1) and the first column is an index.
        - val_size: float (default: 0.2)
            Proportion of the dataset to include in the validation split.
        - random_state: int (default: 1)
            Random seed for reproducibility.

        Returns:
        - X_train, X_val, y_train, y_val: numpy arrays
            Training and validation splits.
        """

        # Set random seed for reproducibility
        np.random.seed(random_state)

        # Create shuffled indices
        shuffled_indices = np.arange(len(y))
        np.random.shuffle(shuffled_indices)

        # Determine indices for label 1 and -1
        label_1_mask = y[shuffled_indices, 1] == 1
        label_minus_1_mask = ~label_1_mask

        # Calculate split index for both sets of indices
        split_idx_1 = int(val_size * np.sum(label_1_mask))
        split_idx_minus_1 = int(val_size * np.sum(label_minus_1_mask))

        # Split the indices
        val_indices = np.concatenate(
            [
                shuffled_indices[label_1_mask][:split_idx_1],
                shuffled_indices[label_minus_1_mask][:split_idx_minus_1],
            ]
        )
        train_indices = np.concatenate(
            [
                shuffled_indices[label_1_mask][split_idx_1:],
                shuffled_indices[label_minus_1_mask][split_idx_minus_1:],
            ]
        )

        # Split the data using the indices
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        return X_train, X_val, y_train, y_val

    def _standardize(self, X):
        """Standardize continuous columns."""

        X[:, self.continuous_cols] = (
            X[:, self.continuous_cols] - self.train_mean
        ) / self.train_std
        return X

    def _one_hot_encode(self, X):
        """
        Generate one-hot encoded columns for self.nominal_ordinal_cols in a numpy array,
        and drop the initial columns. Should work exactly the same as pandas.get_dummies(..., dummy_na=True).

        Parameters:
        - X: numpy array
            Array containing the categorical data to be transformed
        """

        # List to hold all one-hot encoded columns
        all_encoded_cols = []

        for col_idx in self.nominal_ordinal_cols:
            unique_values = np.unique(X[:, col_idx])

            # Create a matrix where each column corresponds to one unique value
            comparison_matrix = (
                X[:, col_idx][:, np.newaxis] == unique_values
            )  # broadcasting

            # Handle NaNs
            nan_column = np.isnan(X[:, col_idx]).astype(int)[:, np.newaxis]
            comparison_matrix = np.hstack([comparison_matrix, nan_column])

            all_encoded_cols.append(comparison_matrix)

        # Concatenate the one-hot encoded columns
        encoded_data = np.hstack(all_encoded_cols)

        # Remove the original nominal/ordinal columns
        remaining_data = np.delete(X, self.nominal_ordinal_cols, axis=1)

        # Combine the remaining data with the one-hot encoded data
        transformed_X = np.hstack([remaining_data, encoded_data])

        return transformed_X

    def _fill_na(self, X):
        """
        Fill NaN values in continuous columns of a numpy array with the mean of that column.

        Parameters:
        - X: numpy array
            The input array with potential NaN values.

        Returns:
        - numpy array with NaN values in continuous columns filled.
        """

        for idx, col in enumerate(self.continuous_cols):
            nan_indices = np.isnan(X[:, col])
            X[nan_indices, col] = self.train_mean[idx]

        return X
