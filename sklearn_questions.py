"""Assignment 3 – scikit-learn API.

This module implements:
- A KNearestNeighbors classifier compatible with scikit-learn
- A MonthlySplit cross-validator based on datetime data
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        """Initialize the classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of nearest neighbors used for voting.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the nearest neighbors classifier."""
        X, y = validate_data(self, X, y)

        if type_of_target(y) == "continuous":
            raise ValueError("continuous target is not supported")

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        check_is_fitted(self, attributes=["X_", "y_", "classes_"])
        X = validate_data(self, X, reset=False)

        distances = pairwise_distances(X, self.X_)
        neighbors_idx = np.argsort(distances, axis=1)[:, : self.n_neighbors]

        y_pred = np.empty(X.shape[0], dtype=self.y_.dtype)
        for i, idx in enumerate(neighbors_idx):
            labels = self.y_[idx]
            values, counts = np.unique(labels, return_counts=True)
            y_pred[i] = values[np.argmax(counts)]

        return y_pred

    def score(self, X, y):
        """Return accuracy on the given test data."""
        return np.mean(self.predict(X) == y)


class MonthlySplit(BaseCrossValidator):
    """Cross-validator based on successive monthly splits.

    Behavior
    --------
    - If time_col == "index": expanding window
      train = all months before the test month
      test  = the test month
    - If time_col != "index": rolling window
      train = previous month only
      test  = next month only
    """

    def __init__(self, time_col="index"):
        """Initialize the splitter.

        Parameters
        ----------
        time_col : str, default="index"
            Where to read the datetime information from. Use "index" to use
            X.index, otherwise use X[time_col] from a DataFrame.
        """
        self.time_col = time_col

    def __repr__(self):
        """Return string representation."""
        return f"MonthlySplit(time_col='{self.time_col}')"

    def _get_datetime(self, X):
        """Extract datetime index or datetime column from X."""
        if isinstance(X, pd.DataFrame):
            if self.time_col == "index":
                time = X.index
            else:
                if self.time_col not in X.columns:
                    raise ValueError("datetime")
                time = X[self.time_col]
        elif isinstance(X, pd.Series):
            # Only sensible interpretation for a Series is to use its index
            if self.time_col != "index":
                raise TypeError("unsupported Type Series")
            time = X.index
        else:
            raise TypeError(f"unsupported Type {type(X).__name__}")

        # Convert to DatetimeIndex and validate dtype
        if isinstance(time, pd.Series):
            if not pd.api.types.is_datetime64_any_dtype(time):
                raise ValueError("datetime")
            time = pd.DatetimeIndex(time)

        if not isinstance(time, pd.DatetimeIndex):
            raise ValueError("datetime")

        return time

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator."""
        time = self._get_datetime(X)
        months = time.to_period("M")
        return max(len(months.unique()) - 1, 0)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        time = self._get_datetime(X)

        # Sort by time (handles shuffled X). We return indices into original X.
        order = np.argsort(time.values)
        time_sorted = time.values[order]
        months_sorted = pd.PeriodIndex(time_sorted, freq="M")
        unique_months = np.sort(months_sorted.unique())

        if len(unique_months) <= 1:
            return

        if self.time_col == "index":
            # Expanding window: all months before current test month
            for test_month in unique_months[1:]:
                train_idx = order[months_sorted < test_month]
                test_idx = order[months_sorted == test_month]
                yield train_idx, test_idx
        else:
            # Rolling window: previous month -> next month
            for prev_month, curr_month in zip(unique_months[:-1], unique_months[1:]):
                train_idx = order[months_sorted == prev_month]
                test_idx = order[months_sorted == curr_month]
                yield train_idx, test_idx
