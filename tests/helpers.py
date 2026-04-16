"""
Shared helper functions for the testing suite.
"""
import numpy as np
from sklearn.model_selection import train_test_split

def make_binary_data(n=100, n_features=5, seed=42):
    """Small reproducible binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    return X, y


def split(X, y, val_frac=0.2, seed=42):
    X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=val_frac, random_state=seed)
    return X_tr, X_v, y_tr, y_v