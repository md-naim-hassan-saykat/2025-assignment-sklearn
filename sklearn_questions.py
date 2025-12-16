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
    def __init__(self, time_col="index"):
        self.time_col = time_col

    def __repr__(self):
        return f"MonthlySplit(time_col='{self.time_col}')"

    def _get_datetime(self, X):
        if isinstance(X, pd.Series):
            time = X.index
        elif isinstance(X, pd.DataFrame):
            if self.time_col == "index":
                time = X.index
            else:
                if self.time_col not in X.columns:
                    raise ValueError("datetime")
                time = X[self.time_col]
        else:
            raise ValueError("datetime")

        if isinstance(time, pd.Series):
            if not pd.api.types.is_datetime64_any_dtype(time):
                raise ValueError("datetime")
            time = pd.DatetimeIndex(time)

        if not isinstance(time, pd.DatetimeIndex):
            raise ValueError("datetime")

        return time

    def split(self, X, y=None, groups=None):
    time = self._get_datetime(X)

    # Sort by time (important if shuffled)
    order = np.argsort(time.values)
    time_sorted = time.values[order]

    months = pd.PeriodIndex(time_sorted, freq="M")
    unique_months = np.sort(months.unique())

    if self.time_col == "index":
        # CUMULATIVE / EXPANDING window
        for test_month in unique_months[1:]:
            train_idx = order[months < test_month]
            test_idx = order[months == test_month]
            yield train_idx, test_idx
    else:
        # ROLLING one-month window
        for prev_month, curr_month in zip(
            unique_months[:-1], unique_months[1:]
        ):
            train_idx = order[months == prev_month]
            test_idx = order[months == curr_month]
            yield train_idx, test_idx
