class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the nearest neighbors classifier."""
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            ensure_min_samples=1,
            ensure_min_features=1,
        )

        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            ensure_min_samples=1,
        )

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
        check_is_fitted(self)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Cross-validator based on successive monthly splits."""

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def _get_datetime_series(self, X):
        """Extract datetime series from DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.time_col == 'index':
            time = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError("time_col not found in X")
            time = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(time):
            raise ValueError("time column must be datetime")

        return pd.to_datetime(time)

    def get_n_splits(self, X, y=None, groups=None):
        """Return number of splitting iterations."""
        time = self._get_datetime_series(X)
        months = time.to_period("M")
        return max(len(months.unique()) - 1, 0)

    def split(self, X, y=None, groups=None):
        """Generate train/test indices."""
        time = self._get_datetime_series(X)
        months = time.to_period("M")
        unique_months = months.unique().sort_values()

        for m_train, m_test in zip(unique_months[:-1], unique_months[1:]):
            idx_train = np.where(months == m_train)[0]
            idx_test = np.where(months == m_test)[0]
            yield idx_train, idx_test
