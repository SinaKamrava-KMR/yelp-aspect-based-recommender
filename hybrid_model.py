from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from text_processor import CANONICAL_ASPECTS, ProfileMatrices


@dataclass
class TrainingData:
    ratings: csr_matrix
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    user_review_counts: dict[str, int]
    item_review_counts: dict[str, int]
    item_mean_ratings: dict[str, float]
    seen_by_user: dict[str, set[str]]
    global_mean: float
    row_count: int

    @property
    def catalogue(self) -> tuple[str, ...]:
        return tuple(sorted(self.item_to_index))


def load_training_data(path: str | Path) -> TrainingData:
    """Build a sparse rating matrix using mean ratings for duplicate pairs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    pair_sum: Counter[tuple[str, str]] = Counter()
    pair_count: Counter[tuple[str, str]] = Counter()
    user_review_counts: Counter[str] = Counter()
    item_review_counts: Counter[str] = Counter()
    item_rating_sums: Counter[str] = Counter()
    seen: dict[str, set[str]] = defaultdict(set)
    total_rating = 0.0
    row_count = 0

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"review_id", "user_id", "business_id", "stars", "date", "text"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"Unexpected training CSV header: {reader.fieldnames}")
        for row in reader:
            try:
                rating = float(row["stars"])
            except ValueError as exc:
                raise ValueError(f"Invalid training rating: {row['stars']!r}") from exc
            if not np.isfinite(rating) or not 1.0 <= rating <= 5.0:
                raise ValueError(f"Training rating out of range: {rating}")
            user_id, item_id = row["user_id"], row["business_id"]
            pair = (user_id, item_id)
            pair_sum[pair] += rating
            pair_count[pair] += 1
            user_review_counts[user_id] += 1
            item_review_counts[item_id] += 1
            item_rating_sums[item_id] += rating
            seen[user_id].add(item_id)
            total_rating += rating
            row_count += 1
    if row_count == 0:
        raise ValueError("Training reviews are empty")

    users = sorted(user_review_counts)
    items = sorted(item_review_counts)
    user_to_index = {value: index for index, value in enumerate(users)}
    item_to_index = {value: index for index, value in enumerate(items)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (user_id, item_id), count in sorted(pair_count.items()):
        rows.append(user_to_index[user_id])
        columns.append(item_to_index[item_id])
        values.append(pair_sum[(user_id, item_id)] / count)
    ratings = csr_matrix(
        (values, (rows, columns)),
        shape=(len(users), len(items)),
        dtype=np.float64,
    )
    item_means = {
        item_id: item_rating_sums[item_id] / item_review_counts[item_id]
        for item_id in items
    }
    return TrainingData(
        ratings=ratings,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        user_review_counts=dict(user_review_counts),
        item_review_counts=dict(item_review_counts),
        item_mean_ratings=item_means,
        seen_by_user=dict(seen),
        global_mean=total_rating / row_count,
        row_count=row_count,
    )


def save_training_data(cache_dir: str | Path, training: TrainingData) -> None:
    """Persist the streamed training aggregation for fast, safe resume."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ratings_tmp = cache_dir / "ratings.tmp.npz"
    save_npz(ratings_tmp, training.ratings, compressed=True)
    users = [""] * len(training.user_to_index)
    for user_id, index in training.user_to_index.items():
        users[index] = user_id
    items = [""] * len(training.item_to_index)
    for item_id, index in training.item_to_index.items():
        items[index] = item_id
    metadata = {
        "user_ids": users,
        "item_ids": items,
        "user_review_counts": [training.user_review_counts[user_id] for user_id in users],
        "item_review_counts": [training.item_review_counts[item_id] for item_id in items],
        "item_mean_ratings": [training.item_mean_ratings[item_id] for item_id in items],
        "global_mean": training.global_mean,
        "row_count": training.row_count,
    }
    metadata_tmp = cache_dir / "training_metadata.json.tmp"
    metadata_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    ratings_tmp.replace(cache_dir / "ratings.npz")
    metadata_tmp.replace(cache_dir / "training_metadata.json")


def load_cached_training_data(cache_dir: str | Path) -> TrainingData:
    cache_dir = Path(cache_dir)
    metadata = json.loads(
        (cache_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    ratings = load_npz(cache_dir / "ratings.npz").tocsr()
    users = metadata["user_ids"]
    items = metadata["item_ids"]
    if ratings.shape != (len(users), len(items)):
        raise ValueError("Cached rating matrix shape does not match its index maps")
    user_to_index = {value: index for index, value in enumerate(users)}
    item_to_index = {value: index for index, value in enumerate(items)}
    seen: dict[str, set[str]] = {}
    for user_index, user_id in enumerate(users):
        start, end = ratings.indptr[user_index : user_index + 2]
        seen[user_id] = {items[item_index] for item_index in ratings.indices[start:end]}
    return TrainingData(
        ratings=ratings,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        user_review_counts=dict(zip(users, metadata["user_review_counts"])),
        item_review_counts=dict(zip(items, metadata["item_review_counts"])),
        item_mean_ratings=dict(zip(items, metadata["item_mean_ratings"])),
        seen_by_user=seen,
        global_mean=float(metadata["global_mean"]),
        row_count=int(metadata["row_count"]),
    )


@dataclass
class BiasedMatrixFactorization:
    factors: int = 32
    epochs: int = 20
    learning_rate: float = 0.01
    regularization: float = 0.05
    seed: int = 42
    global_mean: float = 0.0
    user_bias: np.ndarray | None = None
    item_bias: np.ndarray | None = None
    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    best_epoch: int | None = None
    validation_history: list[float] = field(default_factory=list)

    def fit(
        self,
        ratings: csr_matrix,
        *,
        validation: Iterable[tuple[int, int, float]] | None = None,
        patience: int | None = None,
        min_delta: float = 1e-4,
    ) -> "BiasedMatrixFactorization":
        if self.factors < 1 or self.epochs < 1:
            raise ValueError("factors and epochs must be positive")
        if self.learning_rate <= 0 or self.regularization < 0:
            raise ValueError("Invalid optimizer hyperparameters")
        if patience is not None and patience < 1:
            raise ValueError("patience must be positive or None")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        coo = ratings.tocoo()
        if coo.nnz == 0:
            raise ValueError("Cannot fit matrix factorization without ratings")
        order = np.lexsort((coo.col, coo.row))
        rows = coo.row[order]
        columns = coo.col[order]
        values = coo.data[order].astype(np.float64, copy=False)
        rng = np.random.default_rng(self.seed)
        self.global_mean = float(values.mean())
        self.user_bias = np.zeros(ratings.shape[0], dtype=np.float64)
        self.item_bias = np.zeros(ratings.shape[1], dtype=np.float64)
        self.user_factors = rng.normal(0.0, 0.05, (ratings.shape[0], self.factors))
        self.item_factors = rng.normal(0.0, 0.05, (ratings.shape[1], self.factors))
        validation_rows = list(validation or ())
        self.validation_history = []
        best_rmse = float("inf")
        best_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        stale_epochs = 0

        for epoch in range(1, self.epochs + 1):
            for index in rng.permutation(len(values)):
                user_index = rows[index]
                item_index = columns[index]
                rating = values[index]
                prediction = (
                    self.global_mean
                    + self.user_bias[user_index]
                    + self.item_bias[item_index]
                    + float(
                        np.dot(
                            self.user_factors[user_index],
                            self.item_factors[item_index],
                        )
                    )
                )
                error = rating - prediction
                self.user_bias[user_index] += self.learning_rate * (
                    error - self.regularization * self.user_bias[user_index]
                )
                self.item_bias[item_index] += self.learning_rate * (
                    error - self.regularization * self.item_bias[item_index]
                )
                old_user = self.user_factors[user_index].copy()
                self.user_factors[user_index] += self.learning_rate * (
                    error * self.item_factors[item_index]
                    - self.regularization * self.user_factors[user_index]
                )
                self.item_factors[item_index] += self.learning_rate * (
                    error * old_user
                    - self.regularization * self.item_factors[item_index]
                )
            if validation_rows:
                squared_error = 0.0
                for user_index, item_index, target in validation_rows:
                    prediction = self.predict(user_index, item_index)
                    squared_error += (prediction - target) ** 2
                current_rmse = float(np.sqrt(squared_error / len(validation_rows)))
                self.validation_history.append(current_rmse)
                if current_rmse < best_rmse - min_delta:
                    best_rmse = current_rmse
                    self.best_epoch = epoch
                    best_arrays = (
                        self.user_bias.copy(),
                        self.item_bias.copy(),
                        self.user_factors.copy(),
                        self.item_factors.copy(),
                    )
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if patience is not None and stale_epochs >= patience:
                    break
        if best_arrays is not None:
            (
                self.user_bias,
                self.item_bias,
                self.user_factors,
                self.item_factors,
            ) = best_arrays
        elif not validation_rows:
            self.best_epoch = self.epochs
        return self

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.user_bias,
                self.item_bias,
                self.user_factors,
                self.item_factors,
            )
        ):
            raise RuntimeError("Matrix-factorization model is not fitted")

    def predict(self, user_index: int | None, item_index: int | None) -> float:
        self._check_fitted()
        assert self.user_bias is not None and self.item_bias is not None
        assert self.user_factors is not None and self.item_factors is not None
        value = self.global_mean
        if user_index is not None:
            value += float(self.user_bias[user_index])
        if item_index is not None:
            value += float(self.item_bias[item_index])
        if user_index is not None and item_index is not None:
            value += float(
                np.dot(self.user_factors[user_index], self.item_factors[item_index])
            )
        return value


def save_matrix_factorization(
    cache_dir: str | Path,
    model: BiasedMatrixFactorization,
) -> None:
    model._check_fitted()
    assert model.user_bias is not None and model.item_bias is not None
    assert model.user_factors is not None and model.item_factors is not None
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_tmp = cache_dir / "mf_arrays.tmp.npz"
    np.savez_compressed(
        arrays_tmp,
        user_bias=model.user_bias,
        item_bias=model.item_bias,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
    )
    metadata_tmp = cache_dir / "mf_metadata.json.tmp"
    metadata_tmp.write_text(
        json.dumps(
            {
                "factors": model.factors,
                "epochs": model.epochs,
                "learning_rate": model.learning_rate,
                "regularization": model.regularization,
                "seed": model.seed,
                "global_mean": model.global_mean,
                "best_epoch": model.best_epoch,
                "validation_history": model.validation_history,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    arrays_tmp.replace(cache_dir / "mf_arrays.npz")
    metadata_tmp.replace(cache_dir / "mf_metadata.json")


def load_matrix_factorization(cache_dir: str | Path) -> BiasedMatrixFactorization:
    cache_dir = Path(cache_dir)
    metadata = json.loads((cache_dir / "mf_metadata.json").read_text(encoding="utf-8"))
    with np.load(cache_dir / "mf_arrays.npz", allow_pickle=False) as arrays:
        model = BiasedMatrixFactorization(
            factors=int(metadata["factors"]),
            epochs=int(metadata["epochs"]),
            learning_rate=float(metadata["learning_rate"]),
            regularization=float(metadata["regularization"]),
            seed=int(metadata["seed"]),
            global_mean=float(metadata["global_mean"]),
            user_bias=arrays["user_bias"].copy(),
            item_bias=arrays["item_bias"].copy(),
            user_factors=arrays["user_factors"].copy(),
            item_factors=arrays["item_factors"].copy(),
            best_epoch=(
                None if metadata.get("best_epoch") is None else int(metadata["best_epoch"])
            ),
            validation_history=[
                float(value) for value in metadata.get("validation_history", [])
            ],
        )
    model._check_fitted()
    return model


def normalize_cf(score: float) -> float:
    return float((np.clip(score, 1.0, 5.0) - 1.0) / 4.0)


def normalize_aspect(score: float) -> float:
    return float((np.clip(score, -1.0, 1.0) + 1.0) / 2.0)


@dataclass(frozen=True)
class ScoreComponents:
    cf_raw: float
    cf_norm: float
    aspect_raw: float
    aspect_norm: float
    hybrid: float


@dataclass
class RecommendationResult:
    rows: list[dict[str, object]]
    fallback: bool
    reason: str
    max_hybrid_score: float | None


class HybridRecommender:
    def __init__(
        self,
        training: TrainingData,
        matrix_factorization: BiasedMatrixFactorization,
        profiles: ProfileMatrices,
        bpr_model: Any | None = None,
    ) -> None:
        self.training = training
        self.mf = matrix_factorization
        self.profiles = profiles
        self.bpr_model = bpr_model
        self._max_popularity = max(training.item_review_counts.values())
        self._catalogue = training.catalogue
        self._catalogue_positions = {
            item_id: position for position, item_id in enumerate(self._catalogue)
        }
        self._catalogue_item_indices = np.asarray(
            [training.item_to_index[item_id] for item_id in self._catalogue],
            dtype=np.int64,
        )
        self._item_profile_vectors = np.zeros(
            (len(self._catalogue), len(CANONICAL_ASPECTS)), dtype=np.float64
        )
        profile_matrix = profiles.item_aspects.toarray()
        for position, item_id in enumerate(self._catalogue):
            profile_index = profiles.item_to_index.get(item_id)
            if profile_index is not None:
                self._item_profile_vectors[position] = profile_matrix[profile_index]
        self._popularity = np.asarray(
            [training.item_review_counts[item_id] for item_id in self._catalogue],
            dtype=np.float64,
        )
        self._popularity_norm = np.log1p(self._popularity) / np.log1p(
            self._max_popularity
        )

    def score(
        self,
        user_id: str,
        business_id: str,
        *,
        alpha: float,
    ) -> ScoreComponents:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        user_index = self.training.user_to_index.get(user_id)
        item_index = self.training.item_to_index.get(business_id)
        cf_raw = self.mf.predict(user_index, item_index)
        user_vector = self.profiles.user_vector(user_id)
        item_vector = self.profiles.item_vector(business_id)
        aspect_raw = float(np.dot(user_vector, item_vector))
        cf_norm = normalize_cf(cf_raw)
        aspect_norm = normalize_aspect(aspect_raw)
        hybrid = alpha * cf_norm + (1.0 - alpha) * aspect_norm
        return ScoreComponents(cf_raw, cf_norm, aspect_raw, aspect_norm, hybrid)

    def score_mode(
        self,
        user_id: str,
        business_id: str,
        *,
        alpha: float = 0.7,
        mode: str = "hybrid",
        popularity_weight: float = 0.15,
        adaptive_cf_k: float = 10.0,
        adaptive_aspect_k: float = 5.0,
    ) -> float:
        """Return a normalized point score for RMSE-compatible advanced models."""
        components = self.score(user_id, business_id, alpha=alpha)
        item_position = self._catalogue_positions.get(business_id)
        popularity = (
            float(self._popularity_norm[item_position])
            if item_position is not None
            else 0.0
        )
        if mode == "hybrid":
            return components.hybrid
        if mode == "popularity":
            return popularity
        if mode == "popularity_hybrid":
            return (
                (1.0 - popularity_weight) * components.hybrid
                + popularity_weight * popularity
            )
        if mode == "adaptive":
            interactions = float(self.training.user_review_counts.get(user_id, 0))
            evidence = self.profiles.user_evidence(user_id)
            cf_reliability = interactions / (interactions + adaptive_cf_k)
            aspect_reliability = evidence / (evidence + adaptive_aspect_k)
            cf_weight = alpha * cf_reliability
            aspect_weight = (1.0 - alpha) * aspect_reliability
            pop_weight = popularity_weight + 1.0 - max(
                cf_reliability, aspect_reliability
            )
            total = cf_weight + aspect_weight + pop_weight
            return (
                cf_weight * components.cf_norm
                + aspect_weight * components.aspect_norm
                + pop_weight * popularity
            ) / total
        if mode == "bpr":
            if self.bpr_model is None:
                raise RuntimeError("BPR mode requested before fitting/loading a BPR model")
            user_index = self.training.user_to_index.get(user_id)
            item_index = self.training.item_to_index.get(business_id)
            if user_index is None or item_index is None:
                return popularity
            raw = float(
                self.bpr_model.score_all(
                    user_index, np.asarray([item_index], dtype=np.int64)
                )[0]
            )
            return float(1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0))))
        raise ValueError(f"Unsupported recommendation mode: {mode}")

    def _component_arrays(self, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute CF/aspect components for the full catalogue in vectorized form."""
        self.mf._check_fitted()
        assert self.mf.item_bias is not None and self.mf.item_factors is not None
        assert self.mf.user_bias is not None and self.mf.user_factors is not None
        item_indices = self._catalogue_item_indices
        cf_raw = self.mf.global_mean + self.mf.item_bias[item_indices]
        user_index = self.training.user_to_index.get(user_id)
        if user_index is not None:
            cf_raw = (
                cf_raw
                + self.mf.user_bias[user_index]
                + self.mf.item_factors[item_indices] @ self.mf.user_factors[user_index]
            )
        cf_norm = (np.clip(cf_raw, 1.0, 5.0) - 1.0) / 4.0

        user_profile_index = self.profiles.user_to_index.get(user_id)
        if user_profile_index is None:
            aspect_raw = np.zeros(len(self._catalogue), dtype=np.float64)
        else:
            user_vector = self.profiles.user_aspects.getrow(user_profile_index).toarray().ravel()
            aspect_raw = self._item_profile_vectors @ user_vector
        aspect_norm = (np.clip(aspect_raw, -1.0, 1.0) + 1.0) / 2.0
        return cf_norm, aspect_norm

    def _candidate_positions(self, user_id: str) -> np.ndarray:
        mask = np.ones(len(self._catalogue), dtype=bool)
        for item_id in self.training.seen_by_user.get(user_id, set()):
            position = self._catalogue_positions.get(item_id)
            if position is not None:
                mask[position] = False
        return np.flatnonzero(mask)

    def _bpr_component(self, user_id: str) -> np.ndarray:
        if self.bpr_model is None:
            raise RuntimeError("BPR mode requested before fitting/loading a BPR model")
        user_index = self.training.user_to_index.get(user_id)
        if user_index is None:
            return self._popularity_norm.copy()
        raw = np.asarray(
            self.bpr_model.score_all(user_index, self._catalogue_item_indices),
            dtype=np.float64,
        )
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))

    def _recommend_from_components(
        self,
        user_id: str,
        cf_norm: np.ndarray,
        aspect_norm: np.ndarray,
        *,
        alpha: float,
        top_k: int,
        min_interactions: int,
        threshold: float,
        mode: Literal[
            "hybrid",
            "popularity",
            "popularity_hybrid",
            "adaptive",
            "bpr",
        ],
        apply_abstention: bool,
        popularity_weight: float = 0.15,
        adaptive_cf_k: float = 10.0,
        adaptive_aspect_k: float = 5.0,
    ) -> RecommendationResult:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if top_k < 1 or min_interactions < 1:
            raise ValueError("top_k and min_interactions must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 <= popularity_weight <= 1.0:
            raise ValueError("popularity_weight must be in [0, 1]")
        if adaptive_cf_k <= 0 or adaptive_aspect_k <= 0:
            raise ValueError("adaptive reliability constants must be positive")
        if mode not in {
            "hybrid",
            "popularity",
            "popularity_hybrid",
            "adaptive",
            "bpr",
        }:
            raise ValueError(f"Unsupported recommendation mode: {mode}")
        positions = self._candidate_positions(user_id)
        if not len(positions):
            return RecommendationResult([], False, "no_candidates", None)

        hybrid = alpha * cf_norm + (1.0 - alpha) * aspect_norm
        if mode == "popularity_hybrid":
            ranking_score = (
                (1.0 - popularity_weight) * hybrid
                + popularity_weight * self._popularity_norm
            )
        elif mode == "adaptive":
            interactions = float(self.training.user_review_counts.get(user_id, 0))
            aspect_evidence = self.profiles.user_evidence(user_id)
            cf_reliability = interactions / (interactions + adaptive_cf_k)
            aspect_reliability = aspect_evidence / (
                aspect_evidence + adaptive_aspect_k
            )
            cf_weight = alpha * cf_reliability
            aspect_weight = (1.0 - alpha) * aspect_reliability
            cold_weight = 1.0 - max(cf_reliability, aspect_reliability)
            pop_weight = popularity_weight + cold_weight
            total_weight = cf_weight + aspect_weight + pop_weight
            ranking_score = (
                cf_weight * cf_norm
                + aspect_weight * aspect_norm
                + pop_weight * self._popularity_norm
            ) / total_weight
        elif mode == "bpr":
            ranking_score = self._bpr_component(user_id)
        else:
            ranking_score = hybrid
        max_score = float(np.max(ranking_score[positions]))
        sparse = self.training.user_review_counts.get(user_id, 0) < min_interactions
        low_score = max_score < threshold
        fallback = mode == "popularity" or (apply_abstention and (sparse or low_score))
        if mode == "popularity":
            reason = "popularity_baseline"
        elif mode == "popularity_hybrid":
            reason = "popularity_aware_hybrid"
        elif mode == "adaptive":
            reason = "adaptive_hybrid"
        elif mode == "bpr":
            reason = "bpr_ranking"
        elif sparse:
            reason = "fewer_than_min_interactions"
        elif low_score:
            reason = "max_score_below_threshold"
        else:
            reason = "hybrid_confident"

        candidate_ids = np.asarray(
            [self._catalogue[position] for position in positions], dtype=str
        )
        primary = self._popularity[positions] if fallback else ranking_score[positions]
        local_order = np.lexsort((candidate_ids, -primary))
        selected = positions[local_order[:top_k]]
        rows: list[dict[str, object]] = []
        for rank, position in enumerate(selected, start=1):
            score = (
                self._popularity[position] / self._max_popularity
                if fallback
                else ranking_score[position]
            )
            rows.append(
                {
                    "user_id": user_id,
                    "business_id": self._catalogue[position],
                    "rank": rank,
                    "score": float(score),
                    "is_fallback": fallback,
                }
            )
        return RecommendationResult(rows, fallback, reason, max_score)

    def candidates(self, user_id: str) -> list[str]:
        seen = self.training.seen_by_user.get(user_id, set())
        return [item for item in self.training.catalogue if item not in seen]

    def recommend_user(
        self,
        user_id: str,
        *,
        alpha: float = 0.7,
        top_k: int = 10,
        min_interactions: int = 5,
        threshold: float = 0.3,
        mode: Literal[
            "hybrid",
            "popularity",
            "popularity_hybrid",
            "adaptive",
            "bpr",
        ] = "hybrid",
        apply_abstention: bool = True,
        popularity_weight: float = 0.15,
        adaptive_cf_k: float = 10.0,
        adaptive_aspect_k: float = 5.0,
    ) -> RecommendationResult:
        cf_norm, aspect_norm = self._component_arrays(user_id)
        return self._recommend_from_components(
            user_id,
            cf_norm,
            aspect_norm,
            alpha=alpha,
            top_k=top_k,
            min_interactions=min_interactions,
            threshold=threshold,
            mode=mode,
            apply_abstention=apply_abstention,
            popularity_weight=popularity_weight,
            adaptive_cf_k=adaptive_cf_k,
            adaptive_aspect_k=adaptive_aspect_k,
        )

    def recommend_many(
        self,
        user_ids: Iterable[str],
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], dict[str, RecommendationResult]]:
        rows: list[dict[str, object]] = []
        results: dict[str, RecommendationResult] = {}
        for user_id in sorted(set(user_ids)):
            result = self.recommend_user(user_id, **kwargs)
            rows.extend(result.rows)
            results[user_id] = result
        return rows, results

    def recommend_many_configurations(
        self,
        user_ids: Iterable[str],
        configurations: Mapping[str, Mapping[str, object]],
    ) -> tuple[
        dict[str, list[dict[str, object]]],
        dict[str, dict[str, RecommendationResult]],
    ]:
        """Reuse one vectorized catalogue pass for all baselines/ablations."""
        rows_by_name = {name: [] for name in configurations}
        results_by_name = {name: {} for name in configurations}
        for user_id in sorted(set(user_ids)):
            cf_norm, aspect_norm = self._component_arrays(user_id)
            for name, values in configurations.items():
                result = self._recommend_from_components(
                    user_id,
                    cf_norm,
                    aspect_norm,
                    alpha=float(values.get("alpha", 0.7)),
                    top_k=int(values.get("top_k", 10)),
                    min_interactions=int(values.get("min_interactions", 5)),
                    threshold=float(values.get("threshold", 0.3)),
                    mode=str(values.get("mode", "hybrid")),  # type: ignore[arg-type]
                    apply_abstention=bool(values.get("apply_abstention", True)),
                    popularity_weight=float(values.get("popularity_weight", 0.15)),
                    adaptive_cf_k=float(values.get("adaptive_cf_k", 10.0)),
                    adaptive_aspect_k=float(values.get("adaptive_aspect_k", 5.0)),
                )
                rows_by_name[name].extend(result.rows)
                results_by_name[name][user_id] = result
        return rows_by_name, results_by_name


def write_recommendations(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ("user_id", "business_id", "rank", "score", "is_fallback")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["score"] = f"{float(row['score']):.10f}"
            serialized["is_fallback"] = str(bool(row["is_fallback"])).lower()
            writer.writerow(serialized)
    temporary_path.replace(path)


def read_user_ids(path: str | Path) -> list[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != ["user_id"]:
            raise ValueError(f"Expected a single user_id column in {path}")
        user_ids = [row["user_id"] for row in reader]
    if any(not user_id for user_id in user_ids):
        raise ValueError(f"Blank user_id in {path}")
    return sorted(set(user_ids))
