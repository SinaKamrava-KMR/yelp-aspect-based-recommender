from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from hybrid_model import BiasedMatrixFactorization, TrainingData


def validation_triplets(
    training: TrainingData,
    targets: Mapping[tuple[str, str], float],
) -> list[tuple[int, int, float]]:
    """Convert dev targets to known train-index triplets without fitting on dev."""
    rows: list[tuple[int, int, float]] = []
    for (user_id, item_id), rating in sorted(targets.items()):
        user_index = training.user_to_index.get(user_id)
        item_index = training.item_to_index.get(item_id)
        if user_index is not None and item_index is not None:
            rows.append((user_index, item_index, float(rating)))
    return rows


def tune_matrix_factorization(
    ratings: csr_matrix,
    validation: Sequence[tuple[int, int, float]],
    *,
    factors: Sequence[int],
    learning_rates: Sequence[float],
    regularizations: Sequence[float],
    epochs: Sequence[int],
    seed: int = 42,
    patience: int = 3,
    min_delta: float = 1e-4,
) -> tuple[BiasedMatrixFactorization, dict[str, object]]:
    """Deterministic dev-RMSE grid search with per-run early stopping."""
    if not validation:
        raise ValueError("MF tuning needs at least one known dev user-item target")
    if not all((factors, learning_rates, regularizations, epochs)):
        raise ValueError("MF grids must not be empty")
    trials: list[dict[str, object]] = []
    best_model: BiasedMatrixFactorization | None = None
    best_key: tuple[float, int, float, float, int] | None = None
    for factor_count, learning_rate, regularization, epoch_count in itertools.product(
        factors, learning_rates, regularizations, epochs
    ):
        model = BiasedMatrixFactorization(
            factors=int(factor_count),
            epochs=int(epoch_count),
            learning_rate=float(learning_rate),
            regularization=float(regularization),
            seed=seed,
        ).fit(
            ratings,
            validation=validation,
            patience=patience,
            min_delta=min_delta,
        )
        best_rmse = min(model.validation_history)
        trial = {
            "factors": int(factor_count),
            "learning_rate": float(learning_rate),
            "regularization": float(regularization),
            "max_epochs": int(epoch_count),
            "best_epoch": model.best_epoch,
            "best_dev_rmse": best_rmse,
            "validation_history": model.validation_history,
        }
        trials.append(trial)
        key = (
            best_rmse,
            int(factor_count),
            float(regularization),
            float(learning_rate),
            int(epoch_count),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_model = model
    assert best_model is not None
    report = {
        "selection_metric": "dev_rmse",
        "validation_pairs": len(validation),
        "patience": patience,
        "min_delta": min_delta,
        "best": min(trials, key=lambda value: (
            float(value["best_dev_rmse"]),
            int(value["factors"]),
            float(value["regularization"]),
            float(value["learning_rate"]),
            int(value["max_epochs"]),
        )),
        "trials": trials,
    }
    return best_model, report


@dataclass
class BayesianPersonalizedRanking:
    """Mini-batch BPR-MF trained only from implicit train interactions."""

    factors: int = 64
    epochs: int = 10
    learning_rate: float = 0.03
    regularization: float = 0.01
    batch_size: int = 4096
    seed: int = 42
    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    loss_history: list[float] = field(default_factory=list)

    def fit(self, interactions: csr_matrix) -> "BayesianPersonalizedRanking":
        if self.factors < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("BPR factors, epochs, and batch_size must be positive")
        if self.learning_rate <= 0 or self.regularization < 0:
            raise ValueError("Invalid BPR optimizer hyperparameters")
        binary = interactions.copy().tocsr()
        binary.data = np.ones_like(binary.data)
        coo = binary.tocoo()
        if coo.nnz == 0 or binary.shape[1] < 2:
            raise ValueError("BPR needs interactions with at least two catalogue items")
        eligible = np.asarray(
            [binary.indptr[user + 1] - binary.indptr[user] < binary.shape[1] for user in coo.row],
            dtype=bool,
        )
        positive_users = coo.row[eligible].astype(np.int64, copy=False)
        positive_items = coo.col[eligible].astype(np.int64, copy=False)
        if not len(positive_users):
            raise ValueError("Every BPR user has interacted with the full catalogue")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.factors)
        self.user_factors = rng.normal(0.0, scale, (binary.shape[0], self.factors))
        self.item_factors = rng.normal(0.0, scale, (binary.shape[1], self.factors))
        seen = [
            set(binary.indices[binary.indptr[user] : binary.indptr[user + 1]])
            for user in range(binary.shape[0])
        ]
        self.loss_history = []

        for _epoch in range(self.epochs):
            order = rng.permutation(len(positive_users))
            epoch_loss = 0.0
            sample_count = 0
            for start in range(0, len(order), self.batch_size):
                selection = order[start : start + self.batch_size]
                users = positive_users[selection]
                positives = positive_items[selection]
                negatives = rng.integers(0, binary.shape[1], size=len(selection))
                for _attempt in range(20):
                    invalid = np.fromiter(
                        (negative in seen[user] for user, negative in zip(users, negatives)),
                        dtype=bool,
                        count=len(users),
                    )
                    if not invalid.any():
                        break
                    negatives[invalid] = rng.integers(0, binary.shape[1], size=int(invalid.sum()))
                for index in np.flatnonzero(invalid):
                    user = int(users[index])
                    candidate = 0
                    while candidate in seen[user]:
                        candidate += 1
                    negatives[index] = candidate

                assert self.user_factors is not None and self.item_factors is not None
                user_vectors = self.user_factors[users].copy()
                positive_vectors = self.item_factors[positives].copy()
                negative_vectors = self.item_factors[negatives].copy()
                margins = np.sum(user_vectors * (positive_vectors - negative_vectors), axis=1)
                gradient = 1.0 / (1.0 + np.exp(np.clip(margins, -30.0, 30.0)))
                epoch_loss += float(np.logaddexp(0.0, -margins).sum())
                sample_count += len(margins)

                user_updates = self.learning_rate * (
                    gradient[:, None] * (positive_vectors - negative_vectors)
                    - self.regularization * user_vectors
                )
                positive_updates = self.learning_rate * (
                    gradient[:, None] * user_vectors
                    - self.regularization * positive_vectors
                )
                negative_updates = self.learning_rate * (
                    -gradient[:, None] * user_vectors
                    - self.regularization * negative_vectors
                )
                np.add.at(self.user_factors, users, user_updates)
                np.add.at(self.item_factors, positives, positive_updates)
                np.add.at(self.item_factors, negatives, negative_updates)
            self.loss_history.append(epoch_loss / sample_count)
        return self

    def _check_fitted(self) -> None:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("BPR model is not fitted")

    def score_all(self, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.user_factors is not None and self.item_factors is not None
        return self.item_factors[item_indices] @ self.user_factors[user_index]


def save_bpr(path: str | Path, model: BayesianPersonalizedRanking) -> None:
    model._check_fitted()
    assert model.user_factors is not None and model.item_factors is not None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays_tmp = path / "bpr_arrays.tmp.npz"
    np.savez_compressed(
        arrays_tmp,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
    )
    metadata_tmp = path / "bpr_metadata.json.tmp"
    metadata_tmp.write_text(
        json.dumps(
            {
                "factors": model.factors,
                "epochs": model.epochs,
                "learning_rate": model.learning_rate,
                "regularization": model.regularization,
                "batch_size": model.batch_size,
                "seed": model.seed,
                "loss_history": model.loss_history,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    arrays_tmp.replace(path / "bpr_arrays.npz")
    metadata_tmp.replace(path / "bpr_metadata.json")


def load_bpr(path: str | Path) -> BayesianPersonalizedRanking:
    path = Path(path)
    metadata = json.loads((path / "bpr_metadata.json").read_text(encoding="utf-8"))
    with np.load(path / "bpr_arrays.npz", allow_pickle=False) as arrays:
        model = BayesianPersonalizedRanking(
            factors=int(metadata["factors"]),
            epochs=int(metadata["epochs"]),
            learning_rate=float(metadata["learning_rate"]),
            regularization=float(metadata["regularization"]),
            batch_size=int(metadata["batch_size"]),
            seed=int(metadata["seed"]),
            user_factors=arrays["user_factors"].copy(),
            item_factors=arrays["item_factors"].copy(),
            loss_history=[float(value) for value in metadata.get("loss_history", [])],
        )
    model._check_fitted()
    return model
