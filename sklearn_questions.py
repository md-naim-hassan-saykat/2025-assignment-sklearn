"""Assignment 3 – scikit-learn API.

This module implements:
- A KNearestNeighbors classifier compatible with scikit-learn
- A MonthlySplit cross-validator based on datetime data
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        if type_of_target(y) == "continuous":
            raise ValueError("continuous target is not supported")

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
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
        return np.mean(self.predict(X) == y)

class MonthlySplit(BaseCrossValidator):
    """Cross-validator based on successive monthly splits."""

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

    def get_n_splits(self, X, y=None, groups=None):
        time = self._get_datetime(X)
        return max(len(time.to_period("M").unique()) - 1, 0)

    def split(self, X, y=None, groups=None):
        time = self._get_datetime(X)

        order = np.argsort(time.values)
        time_sorted = time.values[order]

        months = pd.PeriodIndex(time_sorted, freq="M")
        unique_months = np.sort(months.unique())

        for test_month in unique_months[1:]:
            train_idx = order[months < test_month]
            test_idx = order[months == test_month]
            yield train_idx, test_idx
