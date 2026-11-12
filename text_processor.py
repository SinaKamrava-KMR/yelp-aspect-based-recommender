from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from sentiment import (
    AspectSentimentResult,
    load_vader_analyzer,
    score_review_aspects_detailed,
    score_review_aspects_mandatory_detailed,
)


CANONICAL_ASPECTS = ("food", "service", "ambiance", "price", "location")
ASPECT_INDEX = {aspect: index for index, aspect in enumerate(CANONICAL_ASPECTS)}


@dataclass(frozen=True)
class AspectMention:
    aspect: str
    root_text: str
    root_lemma: str
    root_index: int
    similarity: float
    head: Any
    similarity_margin: float = 0.0
    mapping_method: str = "spacy"


@dataclass
class ProfileMatrices:
    user_aspects: csr_matrix
    item_aspects: csr_matrix
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    stats: dict[str, Any]
    user_evidence_counts: dict[str, float] = field(default_factory=dict)
    item_evidence_counts: dict[str, float] = field(default_factory=dict)

    def user_vector(self, user_id: str) -> np.ndarray:
        index = self.user_to_index.get(user_id)
        if index is None:
            return np.zeros(len(CANONICAL_ASPECTS), dtype=np.float64)
        return self.user_aspects.getrow(index).toarray().ravel()

    def item_vector(self, business_id: str) -> np.ndarray:
        index = self.item_to_index.get(business_id)
        if index is None:
            return np.zeros(len(CANONICAL_ASPECTS), dtype=np.float64)
        return self.item_aspects.getrow(index).toarray().ravel()

    def user_evidence(self, user_id: str) -> float:
        return float(self.user_evidence_counts.get(user_id, 0.0))

    def item_evidence(self, business_id: str) -> float:
        return float(self.item_evidence_counts.get(business_id, 0.0))


def load_spacy_model(model_name: str = "en_core_web_md") -> Any:
    """Load the required vector model with parser, tagger, and lemmatizer."""
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is not installed; install requirements.txt first") from exc
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is unavailable. Install the supplied "
            "en_core_web_md-3.7.1 wheel."
        ) from exc
    required = {"parser", "tagger", "lemmatizer"}
    missing = sorted(required.difference(nlp.pipe_names))
    if missing:
        raise RuntimeError(f"spaCy model is missing required pipes: {missing}")
    if not nlp.vocab.vectors_length:
        raise RuntimeError("The selected spaCy model has no static word vectors")
    return nlp


def build_seed_tokens(nlp: Any) -> dict[str, Any]:
    seeds: dict[str, Any] = {}
    for aspect in CANONICAL_ASPECTS:
        doc = nlp.make_doc(aspect)
        if len(doc) != 1 or not doc[0].has_vector or doc[0].vector_norm == 0:
            raise RuntimeError(f"Canonical aspect seed has no usable vector: {aspect}")
        seeds[aspect] = doc[0]
    return seeds


def map_token_to_aspect(
    token: Any,
    seed_tokens: Mapping[str, Any],
    *,
    threshold: float = 0.35,
) -> tuple[str, float] | None:
    """Map one noun-chunk root using token-to-token spaCy similarity."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    best_aspect: str | None = None
    best_score = float("-inf")
    for aspect in CANONICAL_ASPECTS:
        score = float(token.similarity(seed_tokens[aspect]))
        # Strict greater-than preserves listed seed order on exact ties.
        if score > best_score:
            best_aspect, best_score = aspect, score
    if best_aspect is None or best_score < threshold:
        return None
    return best_aspect, best_score


def _map_token_with_margin(
    token: Any,
    seed_tokens: Mapping[str, Any],
    *,
    threshold: float,
) -> tuple[str, float, float] | None:
    scored = [
        (aspect, float(token.similarity(seed_tokens[aspect])))
        for aspect in CANONICAL_ASPECTS
    ]
    ranked = sorted(
        enumerate(scored),
        key=lambda value: (-value[1][1], value[0]),
    )
    _position, (best_aspect, best_score) = ranked[0]
    if best_score < threshold:
        return None
    second_score = ranked[1][1][1] if len(ranked) > 1 else -1.0
    return best_aspect, best_score, best_score - second_score


def extract_aspect_mentions(
    doc: Any,
    seed_tokens: Mapping[str, Any] | None,
    *,
    threshold: float = 0.35,
    mapper: Any | None = None,
) -> list[AspectMention]:
    """Enumerate noun chunks in document order and map only their root tokens."""
    if mapper is None and seed_tokens is None:
        raise ValueError("seed_tokens are required when mapper is not supplied")
    mentions: list[AspectMention] = []
    for chunk in doc.noun_chunks:
        token = chunk.root
        if (
            getattr(token, "pos_", "") == "PRON"
            or bool(getattr(token, "is_stop", False))
            or bool(getattr(token, "is_punct", False))
            or (
                mapper is None
                and (
                    not bool(getattr(token, "has_vector", False))
                    or float(getattr(token, "vector_norm", 0.0)) == 0.0
                )
            )
        ):
            continue
        if mapper is None:
            assert seed_tokens is not None
            mapped = _map_token_with_margin(token, seed_tokens, threshold=threshold)
            mapping_method = "spacy"
        else:
            mapped = mapper.map_token(token, threshold=threshold)
            mapping_method = str(getattr(mapper, "name", "custom"))
        if mapped is None:
            continue
        if len(mapped) == 2:
            aspect, similarity = mapped
            margin = 0.0
        else:
            aspect, similarity, margin = mapped
        mentions.append(
            AspectMention(
                aspect=aspect,
                root_text=token.text,
                root_lemma=token.lemma_,
                root_index=token.i,
                similarity=similarity,
                head=token,
                similarity_margin=float(margin),
                mapping_method=mapping_method,
            )
        )
    return mentions


def _text_chunks(text: str, max_chars: int | None) -> list[str]:
    """Split pathological reviews without dropping text or changing row order."""
    if max_chars is None or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        stop = min(start + max_chars, len(text))
        if stop < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, stop)
            if boundary > start:
                stop = boundary + 1
        chunks.append(text[start:stop])
        start = stop
    return chunks


def _iter_training_texts(
    path: Path,
    *,
    max_text_chars: int | None = None,
) -> Iterator[tuple[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"review_id", "user_id", "business_id", "stars", "date", "text"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"Unexpected training CSV header: {reader.fieldnames}")
        for row in reader:
            chunks = _text_chunks(row["text"], max_text_chars)
            for chunk_index, text in enumerate(chunks):
                yield text, {
                    "review_id": row["review_id"],
                    "user_id": row["user_id"],
                    "business_id": row["business_id"],
                    "stars": row["stars"],
                    "date": row["date"],
                    "chunk_index": str(chunk_index),
                    "chunk_count": str(len(chunks)),
                }


def _matrix_from_rows(
    rows: Iterable[tuple[int, int, float]],
    shape: tuple[int, int],
) -> csr_matrix:
    materialized = list(rows)
    if not materialized:
        return csr_matrix(shape, dtype=np.float64)
    row_indices, columns, values = zip(*materialized)
    return csr_matrix((values, (row_indices, columns)), shape=shape, dtype=np.float64)


def _pearson_from_moments(values: list[float]) -> float | None:
    count, sum_x, sum_y, sum_xx, sum_yy, sum_xy = values
    if count < 2:
        return None
    numerator = count * sum_xy - sum_x * sum_y
    denominator = math.sqrt(
        max(count * sum_xx - sum_x * sum_x, 0.0)
        * max(count * sum_yy - sum_y * sum_y, 0.0)
    )
    return numerator / denominator if denominator else None


def build_aspect_profiles(
    train_reviews_path: str | Path,
    *,
    nlp: Any | None = None,
    analyzer: Any | None = None,
    model_name: str = "en_core_web_md",
    threshold: float = 0.35,
    batch_size: int = 1000,
    n_process: int = 4,
    evidence_path: str | Path | None = None,
    evidence_limit: int | None = None,
    progress_every: int = 0,
    aspect_mapper: Any | None = None,
    sentiment_scorer: Any | None = None,
    confidence_weighted: bool = False,
    shrinkage: float = 0.0,
    max_text_chars: int | None = None,
    advanced_sentiment: bool = False,
) -> ProfileMatrices:
    """Parse train text and build sparse U/V profiles with optional denoising."""
    path = Path(train_reviews_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(n_process, bool) or n_process < 1:
        raise ValueError("n_process must be a positive integer")
    if evidence_limit is not None and (
        isinstance(evidence_limit, bool) or evidence_limit < 0
    ):
        raise ValueError("evidence_limit must be non-negative or None")
    if isinstance(progress_every, bool) or progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")
    if max_text_chars is not None and max_text_chars < 1_000:
        raise ValueError("max_text_chars must be at least 1000 or None")
    nlp = nlp or load_spacy_model(model_name)
    if sentiment_scorer is None:
        analyzer = analyzer or load_vader_analyzer()
    seed_tokens = build_seed_tokens(nlp) if aspect_mapper is None else None

    user_mentions: dict[str, Counter[str]] = defaultdict(Counter)
    user_evidence_counts: Counter[str] = Counter()
    item_sentiment_sums: dict[str, Counter[str]] = defaultdict(Counter)
    item_sentiment_weights: dict[str, Counter[str]] = defaultdict(Counter)
    item_evidence_counts: Counter[str] = Counter()
    global_sentiment_sums: Counter[str] = Counter()
    global_sentiment_weights: Counter[str] = Counter()
    mapped_terms: dict[str, Counter[str]] = defaultdict(Counter)
    mapped_similarity: dict[str, Counter[str]] = defaultdict(Counter)
    ambiguous_terms: Counter[str] = Counter()
    correlation_moments: dict[str, list[float]] = {
        aspect: [0.0] * 6 for aspect in CANONICAL_ASPECTS
    }
    all_users: set[str] = set()
    all_items: set[str] = set()
    stats: Counter[str] = Counter()

    evidence_file = None
    if evidence_path is not None:
        evidence_path = Path(evidence_path)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_file = evidence_path.open("w", encoding="utf-8", newline="\n")
    try:
        stream = _iter_training_texts(path, max_text_chars=max_text_chars)
        parsed = nlp.pipe(
            stream,
            as_tuples=True,
            batch_size=batch_size,
            n_process=n_process,
        )
        for doc, metadata in parsed:
            stats["training_chunks_processed"] += 1
            if metadata.get("chunk_index", "0") == "0":
                stats["training_rows_processed"] += 1
            if int(metadata.get("chunk_count", "1")) > 1:
                stats["long_review_chunks_processed"] += 1
            stats["total_tokens_processed"] += len(doc)
            user_id = metadata["user_id"]
            business_id = metadata["business_id"]
            all_users.add(user_id)
            all_items.add(business_id)

            mentions = extract_aspect_mentions(
                doc,
                seed_tokens,
                threshold=threshold,
                mapper=aspect_mapper,
            )
            if mentions:
                stats["review_chunks_with_aspect"] += 1
            for mention in mentions:
                mention_weight = mention.similarity if confidence_weighted else 1.0
                user_mentions[user_id][mention.aspect] += mention_weight
                user_evidence_counts[user_id] += mention_weight
                stats[f"aspect_frequency_{mention.aspect}"] += 1
                lemma = mention.root_lemma.lower()
                mapped_terms[mention.aspect][lemma] += 1
                mapped_similarity[mention.aspect][lemma] += mention.similarity
                if mention.similarity_margin < 0.05:
                    ambiguous_terms[lemma] += 1

            mention_pairs = [(mention.aspect, mention.head) for mention in mentions]
            if sentiment_scorer is None:
                assert analyzer is not None
                if advanced_sentiment:
                    sentiment = score_review_aspects_detailed(mention_pairs, analyzer)
                else:
                    sentiment = score_review_aspects_mandatory_detailed(
                        mention_pairs, analyzer
                    )
            else:
                sentiment = sentiment_scorer.score_review(mention_pairs)
                if not isinstance(sentiment, AspectSentimentResult):
                    raise TypeError("sentiment_scorer.score_review must return AspectSentimentResult")

            mapping_confidence: dict[str, float] = {}
            for aspect in sentiment.scores:
                similarities = [
                    mention.similarity for mention in mentions if mention.aspect == aspect
                ]
                mapping_confidence[aspect] = (
                    sum(similarities) / len(similarities) if similarities else 0.0
                )
            stars = float(metadata["stars"])
            for aspect, score in sentiment.scores.items():
                if confidence_weighted:
                    weight = sentiment.confidence.get(aspect, 0.0) * mapping_confidence[aspect]
                else:
                    weight = 1.0
                if weight > 0.0:
                    item_sentiment_sums[business_id][aspect] += weight * score
                    item_sentiment_weights[business_id][aspect] += weight
                    item_evidence_counts[business_id] += weight
                    global_sentiment_sums[aspect] += weight * score
                    global_sentiment_weights[aspect] += weight
                if abs(score) < 1e-12:
                    stats[f"zero_sentiment_{aspect}"] += 1
                moments = correlation_moments[aspect]
                moments[0] += 1.0
                moments[1] += stars
                moments[2] += score
                moments[3] += stars * stars
                moments[4] += score * score
                moments[5] += stars * score

            if evidence_file is not None and (
                evidence_limit is None
                or stats["evidence_rows_written"] < evidence_limit
            ):
                evidence = {
                    "review_id": metadata["review_id"],
                    "user_id": user_id,
                    "business_id": business_id,
                    "mentions": [
                        {
                            "aspect": mention.aspect,
                            "root": mention.root_text,
                            "similarity": round(mention.similarity, 6),
                            "similarity_margin": round(mention.similarity_margin, 6),
                            "mapping_method": mention.mapping_method,
                        }
                        for mention in mentions
                    ],
                    "opinion_phrases": sentiment.phrases,
                    "aspect_sentiment": {
                        key: round(value, 6) for key, value in sentiment.scores.items()
                    },
                    "sentiment_confidence": {
                        key: round(value, 6)
                        for key, value in sentiment.confidence.items()
                    },
                    "opinion_evidence": sentiment.evidence,
                }
                evidence_file.write(json.dumps(evidence, ensure_ascii=False) + "\n")
                stats["evidence_rows_written"] += 1

            if (
                progress_every
                and metadata.get("chunk_index", "0") == "0"
                and stats["training_rows_processed"] % progress_every == 0
            ):
                print(
                    "[NLP] processed "
                    f"{stats['training_rows_processed']:,} training reviews",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if evidence_file is not None:
            evidence_file.close()

    sorted_users = sorted(all_users)
    sorted_items = sorted(all_items)
    user_to_index = {value: index for index, value in enumerate(sorted_users)}
    item_to_index = {value: index for index, value in enumerate(sorted_items)}

    user_rows: list[tuple[int, int, float]] = []
    for user_id, counts in user_mentions.items():
        total = sum(counts.values())
        if total:
            for aspect in CANONICAL_ASPECTS:
                if counts[aspect]:
                    user_rows.append(
                        (user_to_index[user_id], ASPECT_INDEX[aspect], counts[aspect] / total)
                    )

    global_aspect_mean = {
        aspect: (
            global_sentiment_sums[aspect] / global_sentiment_weights[aspect]
            if global_sentiment_weights[aspect]
            else 0.0
        )
        for aspect in CANONICAL_ASPECTS
    }
    item_rows: list[tuple[int, int, float]] = []
    for business_id, weights in item_sentiment_weights.items():
        for aspect in CANONICAL_ASPECTS:
            if weights[aspect]:
                value = (
                    item_sentiment_sums[business_id][aspect]
                    + shrinkage * global_aspect_mean[aspect]
                ) / (weights[aspect] + shrinkage)
                item_rows.append((item_to_index[business_id], ASPECT_INDEX[aspect], value))

    user_matrix = _matrix_from_rows(user_rows, (len(sorted_users), len(CANONICAL_ASPECTS)))
    item_matrix = _matrix_from_rows(item_rows, (len(sorted_items), len(CANONICAL_ASPECTS)))
    stats["users_in_aspect_matrix"] = len(sorted_users)
    stats["items_in_aspect_matrix"] = len(sorted_items)
    stats["user_aspect_nnz"] = user_matrix.nnz
    stats["item_aspect_nnz"] = item_matrix.nnz
    row_count = stats["training_rows_processed"]
    diagnostics: dict[str, Any] = {
        "review_aspect_coverage": (
            stats["review_chunks_with_aspect"] / stats["training_chunks_processed"]
            if stats["training_chunks_processed"]
            else 0.0
        ),
        "confidence_weighted": bool(confidence_weighted),
        "shrinkage": float(shrinkage),
        "mapping_method": str(getattr(aspect_mapper, "name", "spacy")),
        "sentiment_method": str(getattr(sentiment_scorer, "name", "vader")),
        "advanced_sentiment_rules": bool(advanced_sentiment),
        "global_aspect_mean": global_aspect_mean,
        "star_sentiment_correlation": {
            aspect: _pearson_from_moments(correlation_moments[aspect])
            for aspect in CANONICAL_ASPECTS
        },
        "top_mapped_terms": {
            aspect: [
                {
                    "term": term,
                    "count": int(count),
                    "mean_similarity": mapped_similarity[aspect][term] / count,
                }
                for term, count in mapped_terms[aspect].most_common(25)
            ]
            for aspect in CANONICAL_ASPECTS
        },
        "ambiguous_low_margin_terms": [
            {"term": term, "count": int(count)}
            for term, count in ambiguous_terms.most_common(25)
        ],
        "average_user_evidence": (
            sum(user_evidence_counts.values()) / len(all_users) if all_users else 0.0
        ),
        "average_item_evidence": (
            sum(item_evidence_counts.values()) / len(all_items) if all_items else 0.0
        ),
        "training_rows": int(row_count),
    }
    stats_payload: dict[str, Any] = dict(sorted(stats.items()))
    stats_payload["diagnostics"] = diagnostics
    return ProfileMatrices(
        user_aspects=user_matrix,
        item_aspects=item_matrix,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        stats=stats_payload,
        user_evidence_counts=dict(user_evidence_counts),
        item_evidence_counts=dict(item_evidence_counts),
    )


def save_profile_matrices(cache_dir: str | Path, profiles: ProfileMatrices) -> None:
    """Persist the expensive spaCy/VADER result without using pickle."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    user_tmp = cache_dir / "user_aspects.tmp.npz"
    item_tmp = cache_dir / "item_aspects.tmp.npz"
    save_npz(user_tmp, profiles.user_aspects, compressed=True)
    save_npz(item_tmp, profiles.item_aspects, compressed=True)

    user_ids = [""] * len(profiles.user_to_index)
    for user_id, index in profiles.user_to_index.items():
        user_ids[index] = user_id
    item_ids = [""] * len(profiles.item_to_index)
    for item_id, index in profiles.item_to_index.items():
        item_ids[index] = item_id
    metadata_tmp = cache_dir / "profile_metadata.json.tmp"
    metadata_tmp.write_text(
        json.dumps(
            {
                "user_ids": user_ids,
                "item_ids": item_ids,
                "stats": profiles.stats,
                "user_evidence_counts": [
                    profiles.user_evidence_counts.get(user_id, 0.0)
                    for user_id in user_ids
                ],
                "item_evidence_counts": [
                    profiles.item_evidence_counts.get(item_id, 0.0)
                    for item_id in item_ids
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    user_tmp.replace(cache_dir / "user_aspects.npz")
    item_tmp.replace(cache_dir / "item_aspects.npz")
    metadata_tmp.replace(cache_dir / "profile_metadata.json")


def load_profile_matrices(cache_dir: str | Path) -> ProfileMatrices:
    """Load a profile cache created by :func:`save_profile_matrices`."""
    cache_dir = Path(cache_dir)
    metadata = json.loads(
        (cache_dir / "profile_metadata.json").read_text(encoding="utf-8")
    )
    user_ids = metadata["user_ids"]
    item_ids = metadata["item_ids"]
    user_matrix = load_npz(cache_dir / "user_aspects.npz").tocsr()
    item_matrix = load_npz(cache_dir / "item_aspects.npz").tocsr()
    expected_user_shape = (len(user_ids), len(CANONICAL_ASPECTS))
    expected_item_shape = (len(item_ids), len(CANONICAL_ASPECTS))
    if user_matrix.shape != expected_user_shape or item_matrix.shape != expected_item_shape:
        raise ValueError("Cached aspect-matrix shapes do not match their index maps")
    return ProfileMatrices(
        user_aspects=user_matrix,
        item_aspects=item_matrix,
        user_to_index={value: index for index, value in enumerate(user_ids)},
        item_to_index={value: index for index, value in enumerate(item_ids)},
        stats=metadata["stats"],
        user_evidence_counts=dict(
            zip(user_ids, metadata.get("user_evidence_counts", [0.0] * len(user_ids)))
        ),
        item_evidence_counts=dict(
            zip(item_ids, metadata.get("item_evidence_counts", [0.0] * len(item_ids)))
        ),
    )
