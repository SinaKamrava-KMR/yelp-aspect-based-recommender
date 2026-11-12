from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    path = Path(path)
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if reader.fieldnames != ["user_id", "business_id", "relevance"]:
            raise ValueError(f"Invalid qrels header in {path}: {reader.fieldnames}")
        for row in reader:
            try:
                relevance = int(row["relevance"])
            except ValueError as exc:
                raise ValueError(f"Invalid relevance: {row['relevance']!r}") from exc
            if relevance not in {0, 1}:
                raise ValueError("relevance must be binary")
            user_qrels = qrels.setdefault(row["user_id"], {})
            item = row["business_id"]
            user_qrels[item] = max(user_qrels.get(item, 0), relevance)
    return qrels


def ranking_metrics(
    recommendation_rows: Iterable[Mapping[str, object]],
    qrels: Mapping[str, Mapping[str, int]],
    user_ids: Iterable[str],
    catalogue: Iterable[str],
    *,
    k: int = 10,
    seen_by_user: Mapping[str, set[str]] | None = None,
) -> dict[str, float]:
    """Compute assignment-defined HR@K and binary NDCG@K."""
    if k < 1:
        raise ValueError("k must be positive")
    evaluation_users = sorted(set(user_ids))
    if not evaluation_users:
        return {f"hr@{k}": 0.0, f"ndcg@{k}": 0.0}
    catalogue_set = set(catalogue)
    ranked: dict[str, list[tuple[int, str]]] = {}
    for row in recommendation_rows:
        user_id = str(row["user_id"])
        item_id = str(row["business_id"])
        rank = int(row["rank"])
        if item_id in catalogue_set and rank <= k:
            ranked.setdefault(user_id, []).append((rank, item_id))

    hits = 0.0
    ndcg_total = 0.0
    for user_id in evaluation_users:
        seen = seen_by_user.get(user_id, set()) if seen_by_user is not None else set()
        relevant = {
            item
            for item, label in qrels.get(user_id, {}).items()
            if label == 1 and item in catalogue_set and item not in seen
        }
        recommendations = [
            item for _, item in sorted(ranked.get(user_id, []), key=lambda pair: pair[0])
        ][:k]
        if relevant.intersection(recommendations):
            hits += 1.0
        dcg = sum(
            (1.0 if item in relevant else 0.0) / math.log2(position + 1)
            for position, item in enumerate(recommendations, start=1)
        )
        ideal_length = min(k, len(relevant))
        idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_length + 1))
        ndcg_total += dcg / idcg if idcg else 0.0
    denominator = len(evaluation_users)
    return {f"hr@{k}": hits / denominator, f"ndcg@{k}": ndcg_total / denominator}


def dev_pair_ratings(
    dev_reviews_path: str | Path,
    user_ids: Iterable[str],
    catalogue: Iterable[str],
) -> dict[tuple[str, str], float]:
    """Collapse dev ratings by arithmetic mean for eligible user-item pairs."""
    users = set(user_ids)
    items = set(catalogue)
    sums: Counter[tuple[str, str]] = Counter()
    counts: Counter[tuple[str, str]] = Counter()
    with Path(dev_reviews_path).open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"review_id", "user_id", "business_id", "stars", "date", "text"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"Unexpected dev CSV header: {reader.fieldnames}")
        for row in reader:
            key = (row["user_id"], row["business_id"])
            if key[0] not in users or key[1] not in items:
                continue
            try:
                rating = float(row["stars"])
            except ValueError as exc:
                raise ValueError(f"Invalid dev rating: {row['stars']!r}") from exc
            sums[key] += rating
            counts[key] += 1
    return {key: sums[key] / count for key, count in sorted(counts.items())}


def rmse(
    targets: Mapping[tuple[str, str], float],
    predictor: Callable[[str, str], float],
) -> float:
    if not targets:
        return 0.0
    squared_error = 0.0
    for (user_id, item_id), target in targets.items():
        prediction = float(predictor(user_id, item_id))
        if not math.isfinite(prediction):
            raise ValueError(f"Non-finite prediction for {(user_id, item_id)}")
        squared_error += (prediction - target) ** 2
    return math.sqrt(squared_error / len(targets))


def evaluate(
    recommendation_rows: Iterable[Mapping[str, object]],
    qrels: Mapping[str, Mapping[str, int]],
    user_ids: Iterable[str],
    catalogue: Iterable[str],
    dev_targets: Mapping[tuple[str, str], float],
    predictor: Callable[[str, str], float],
    *,
    k: int = 10,
    seen_by_user: Mapping[str, set[str]] | None = None,
) -> dict[str, float]:
    metrics = ranking_metrics(
        recommendation_rows,
        qrels,
        user_ids,
        catalogue,
        k=k,
        seen_by_user=seen_by_user,
    )
    return {"rmse": rmse(dev_targets, predictor), **metrics}
