"""Traditional machine-learning baselines for multi-output curve regression."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def train_ridge_baseline(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    return model


def train_random_forest_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 80,
    max_depth: int | None = 18,
    random_state: int = 42,
    n_jobs: int = -1,
):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    return model


def train_knn_baseline(x_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 7):
    model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))
    model.fit(x_train, y_train)
    return model


def save_sklearn_model(model, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def load_sklearn_model(path: str | Path):
    return joblib.load(path)
