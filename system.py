from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from data_preprocessor import find_raw_yelp_file, run_phase0
from evaluator import dev_pair_ratings, evaluate, load_qrels
from hybrid_model import (
    BiasedMatrixFactorization,
    HybridRecommender,
    load_cached_training_data,
    load_matrix_factorization,
    load_training_data,
    read_user_ids,
    save_matrix_factorization,
    save_training_data,
    write_recommendations,
)
from text_processor import (
    CANONICAL_ASPECTS,
    build_aspect_profiles,
    load_profile_matrices,
    load_spacy_model,
    save_profile_matrices,
)


# Version 2 deliberately matches the already-computed mandatory checkpoints from
# the original submission.  Advanced artifacts have their own independent key.
CACHE_VERSION = 2
ADVANCED_CACHE_VERSION = 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aspect-Based Hybrid Recommender for Yelp reviews"
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--train-reviews", type=Path, required=True)
    parser.add_argument("--businesses", type=Path, required=True)
    parser.add_argument("--dev-reviews", type=Path, required=True)
    parser.add_argument("--dev-qrels", type=Path, required=True)
    parser.add_argument("--dev-users", type=Path, required=True)
    parser.add_argument("--test-users", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--n-process", type=int, default=4)
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--similarity-threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated Phase 0, NLP-profile, rating, and MF checkpoints.",
    )
    parser.add_argument(
        "--evidence-limit",
        type=int,
        default=200,
        help="Maximum aspect-evidence rows to write (not a required submission file).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Print NLP progress after this many training reviews; 0 disables it.",
    )
    parser.add_argument("--use-transformer", action="store_true")
    parser.add_argument("--enable-advanced-vader", action="store_true")
    parser.add_argument("--run-all-experiments", action="store_true")
    parser.add_argument("--enable-advanced-rankers", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--auto-select", action="store_true")
    parser.add_argument(
        "--alpha-grid", default="0,0.25,0.5,0.7,0.8,0.9,0.95,1"
    )
    parser.add_argument("--min-interactions-grid", default="3,5,10,20")
    parser.add_argument("--threshold-grid", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--confidence-weighted", action="store_true")
    parser.add_argument("--aspect-shrinkage", type=float, default=0.0)
    parser.add_argument("--max-text-chars", type=int, default=20_000)
    parser.add_argument("--enable-word2vec", action="store_true")
    parser.add_argument("--word2vec-vector-size", type=int, default=100)
    parser.add_argument("--word2vec-window", type=int, default=5)
    parser.add_argument("--word2vec-min-count", type=int, default=3)
    parser.add_argument("--word2vec-epochs", type=int, default=5)
    parser.add_argument("--word2vec-workers", type=int, default=1)
    parser.add_argument(
        "--transformer-model",
        default="yangheng/deberta-v3-base-absa-v1.1",
    )
    parser.add_argument("--transformer-device", default="auto")
    parser.add_argument("--transformer-batch-size", type=int, default=16)
    parser.add_argument("--transformer-max-length", type=int, default=256)
    parser.add_argument("--tune-mf", action="store_true")
    parser.add_argument("--mf-factors-grid", default="32,64")
    parser.add_argument("--mf-learning-rate-grid", default="0.005,0.01")
    parser.add_argument("--mf-regularization-grid", default="0.01,0.05")
    parser.add_argument("--mf-epochs-grid", default="20")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--train-bpr", action="store_true")
    parser.add_argument("--bpr-factors", type=int, default=64)
    parser.add_argument("--bpr-epochs", type=int, default=10)
    parser.add_argument("--bpr-learning-rate", type=float, default=0.03)
    parser.add_argument("--bpr-regularization", type=float, default=0.01)
    parser.add_argument("--bpr-batch-size", type=int, default=4096)
    parser.add_argument("--popularity-weight", type=float, default=0.15)
    parser.add_argument("--adaptive-cf-k", type=float, default=10.0)
    parser.add_argument("--adaptive-aspect-k", type=float, default=5.0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if not 0.0 <= args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be in [0, 1]")
    for name in ("n_process", "min_interactions", "top_k", "batch_size", "factors", "epochs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("evidence_limit", "progress_every"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.top_k != 10:
        parser.error("The assignment requires --top-k 10")
    if args.aspect_shrinkage < 0:
        parser.error("--aspect-shrinkage must be non-negative")
    if args.max_text_chars < 1_000:
        parser.error("--max-text-chars must be at least 1000")
    if not 0.0 <= args.popularity_weight <= 1.0:
        parser.error("--popularity-weight must be in [0, 1]")
    for name in (
        "word2vec_vector_size",
        "word2vec_window",
        "word2vec_min_count",
        "word2vec_epochs",
        "word2vec_workers",
        "transformer_batch_size",
        "transformer_max_length",
        "early_stopping_patience",
        "bpr_factors",
        "bpr_epochs",
        "bpr_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "bpr_learning_rate",
        "adaptive_cf_k",
        "adaptive_aspect_k",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.bpr_regularization < 0:
        parser.error("--bpr-regularization must be non-negative")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_float_grid(
    value: str,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    try:
        values = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated numeric grid") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    if minimum is not None and any(item < minimum for item in values):
        raise ValueError(f"{name} values must be >= {minimum}")
    if maximum is not None and any(item > maximum for item in values):
        raise ValueError(f"{name} values must be <= {maximum}")
    return values


def _parse_int_grid(value: str, name: str, *, minimum: int = 1) -> list[int]:
    try:
        values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer grid") from exc
    if not values or any(item < minimum for item in values):
        raise ValueError(f"{name} values must be integers >= {minimum}")
    return values


def _metric_selection_key(
    name: str,
    metrics: Mapping[str, float],
) -> tuple[float, float, float, str]:
    """NDCG first, then HR, then RMSE, with a deterministic name tie-break."""
    return (
        -float(metrics["ndcg@10"]),
        -float(metrics["hr@10"]),
        float(metrics["rmse"]),
        name,
    )


def _log(message: str) -> None:
    print(f"[system] {message}", file=sys.stderr, flush=True)


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_cache_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if actual.get("cache_key") != expected:
        return None
    return actual


def _write_cache_manifest(
    path: Path,
    cache_key: dict[str, Any],
    **extra: Any,
) -> None:
    _write_json(path, {"cache_key": cache_key, **extra})


def _outputs_match(signatures: dict[str, Any]) -> bool:
    try:
        return all(
            _file_signature(Path(value["path"])) == value
            for value in signatures.values()
        )
    except (FileNotFoundError, KeyError, TypeError):
        return False


def _business_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                row = json.loads(line)
                names[row["business_id"]] = row["name"]
    return names


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return max(sum(1 for _ in csv.reader(input_file)) - 1, 0)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _star_predictor(
    recommender: HybridRecommender,
    alpha: float,
) -> Callable[[str, str], float]:
    return lambda user_id, item_id: 1.0 + 4.0 * recommender.score(
        user_id, item_id, alpha=alpha
    ).hybrid


def _model_metrics(
    recommender: HybridRecommender,
    user_ids: list[str],
    qrels: dict[str, dict[str, int]],
    dev_targets: dict[tuple[str, str], float],
    *,
    alpha: float,
    top_k: int,
    min_interactions: int,
    threshold: float,
    mode: str = "hybrid",
    predictor: Callable[[str, str], float] | None = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows, _ = recommender.recommend_many(
        user_ids,
        alpha=alpha,
        top_k=top_k,
        min_interactions=min_interactions,
        threshold=threshold,
        mode=mode,
        apply_abstention=True,
    )
    metrics = evaluate(
        rows,
        qrels,
        user_ids,
        recommender.training.catalogue,
        dev_targets,
        predictor or _star_predictor(recommender, alpha),
        k=top_k,
        seen_by_user=recommender.training.seen_by_user,
    )
    return rows, metrics


def _trace_entry(
    recommender: HybridRecommender,
    user_id: str,
    result: Any,
    business_names: dict[str, str],
    alpha: float,
    threshold: float,
) -> dict[str, Any]:
    user_vector = recommender.profiles.user_vector(user_id)
    preferences = [
        {"aspect": aspect, "score": round(float(user_vector[index]), 6)}
        for index, aspect in enumerate(CANONICAL_ASPECTS)
    ]
    entry: dict[str, Any] = {
        "user_id": user_id,
        "user_train_review_count": recommender.training.user_review_counts.get(user_id, 0),
        "extracted_user_aspect_preferences": preferences,
        "abstain_fallback_triggered": result.fallback,
        "decision_reason": result.reason,
        "max_hybrid_score": (
            None if result.max_hybrid_score is None else round(result.max_hybrid_score, 6)
        ),
        "threshold": threshold,
        "top_recommended_item": None,
    }
    if not result.rows:
        return entry
    business_id = str(result.rows[0]["business_id"])
    components = recommender.score(user_id, business_id, alpha=alpha)
    item_vector = recommender.profiles.item_vector(business_id)
    alignment = []
    for index, aspect in enumerate(CANONICAL_ASPECTS):
        contribution = float(user_vector[index] * item_vector[index])
        label = "positive" if contribution > 0.05 else "negative" if contribution < -0.05 else "neutral"
        alignment.append(
            {
                "aspect": aspect,
                "item_sentiment": round(float(item_vector[index]), 6),
                "weighted_contribution": round(contribution, 6),
                "contribution": label,
            }
        )
    entry["top_recommended_item"] = {
        "business_id": business_id,
        "business_name": business_names.get(business_id, ""),
        "svd_score_norm": round(components.cf_norm, 6),
        "aspect_score_norm": round(components.aspect_norm, 6),
        "final_hybrid_score": round(components.hybrid, 6),
        "output_score": round(float(result.rows[0]["score"]), 6),
        "item_aspect_alignment": alignment,
    }
    return entry


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the mandatory system plus independently named advanced experiments."""
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.run_all_experiments:
        # Transformer remains explicit: on CPU it can take far longer than all
        # other experiments combined.  Passing --use-transformer adds it while
        # reusing every checkpoint produced here.
        args.enable_advanced_vader = True
        args.enable_advanced_rankers = True
        args.sweep = True
        args.auto_select = True
        args.confidence_weighted = True
        args.tune_mf = True
        args.train_bpr = True
        args.enable_word2vec = True
        if args.aspect_shrinkage == 0.0:
            args.aspect_shrinkage = 5.0

    alpha_grid = _parse_float_grid(args.alpha_grid, "--alpha-grid", minimum=0, maximum=1)
    interaction_grid = _parse_int_grid(
        args.min_interactions_grid, "--min-interactions-grid"
    )
    threshold_grid = _parse_float_grid(
        args.threshold_grid, "--threshold-grid", minimum=0, maximum=1
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    phase0_stats: dict[str, int] = {}
    if args.raw_dir is not None:
        business_raw = find_raw_yelp_file(
            args.raw_dir, "yelp_academic_dataset_business.json"
        )
        review_raw = find_raw_yelp_file(
            args.raw_dir, "yelp_academic_dataset_review.json"
        )
        phase0_key = {
            "version": CACHE_VERSION,
            "business_raw": _file_signature(business_raw),
            "review_raw": _file_signature(review_raw),
            "city": "Philadelphia",
            "split_date": "2021-01-01",
            "min_train_rows": 3,
        }
        manifest_path = cache_dir / "phase0_manifest.json"
        manifest = _read_cache_manifest(manifest_path, phase0_key) if args.resume else None
        if manifest is not None and _outputs_match(manifest.get("outputs", {})):
            _log("Phase 0 checkpoint is valid; skipping the 5.34 GB raw scan")
            phase0_stats = {key: int(value) for key, value in manifest.get("stats", {}).items()}
        else:
            _log("Phase 0: streaming/filtering the raw Yelp files")
            phase0_stats = run_phase0(
                args.raw_dir,
                args.businesses,
                args.train_reviews,
                args.dev_reviews,
                args.dev_qrels,
                args.dev_users,
                min_train_rows=3,
            )
            outputs = {
                name: _file_signature(path)
                for name, path in {
                    "businesses": args.businesses,
                    "train_reviews": args.train_reviews,
                    "dev_reviews": args.dev_reviews,
                    "dev_qrels": args.dev_qrels,
                    "dev_users": args.dev_users,
                }.items()
            }
            _write_cache_manifest(
                manifest_path, phase0_key, outputs=outputs, stats=phase0_stats
            )

    for required_path in (
        args.train_reviews,
        args.businesses,
        args.dev_reviews,
        args.dev_qrels,
        args.dev_users,
        args.test_users,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    _log("Loading spaCy model")
    nlp = load_spacy_model("en_core_web_md")
    model_version = str(nlp.meta.get("version", "unknown"))
    if model_version != "3.7.1":
        raise RuntimeError(
            f"Expected en_core_web_md 3.7.1, found {model_version}. Install the supplied wheel."
        )
    training_signature = _file_signature(args.train_reviews)

    mandatory_profile_key = {
        "version": CACHE_VERSION,
        "train_reviews": training_signature,
        "model": "en_core_web_md",
        "model_version": model_version,
        "threshold": args.similarity_threshold,
        "aspects": list(CANONICAL_ASPECTS),
    }
    mandatory_profile_dir = cache_dir / "profiles"
    mandatory_manifest_path = mandatory_profile_dir / "manifest.json"
    mandatory_manifest = (
        _read_cache_manifest(mandatory_manifest_path, mandatory_profile_key)
        if args.resume
        else None
    )
    try:
        profiles = (
            load_profile_matrices(mandatory_profile_dir)
            if mandatory_manifest is not None
            else None
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        profiles = None
    if profiles is None:
        _log("Mandatory NLP profiles: starting assignment-exact spaCy/VADER")
        profiles = build_aspect_profiles(
            args.train_reviews,
            nlp=nlp,
            threshold=args.similarity_threshold,
            batch_size=args.batch_size,
            n_process=args.n_process,
            evidence_path=args.outdir / "aspect_evidence.jsonl",
            evidence_limit=args.evidence_limit,
            progress_every=args.progress_every,
            max_text_chars=args.max_text_chars,
            advanced_sentiment=False,
        )
        save_profile_matrices(mandatory_profile_dir, profiles)
        _write_cache_manifest(mandatory_manifest_path, mandatory_profile_key)
    else:
        _log("Mandatory NLP profile checkpoint is valid; skipping spaCy/VADER")

    training_key = {"version": CACHE_VERSION, "train_reviews": training_signature}
    training_cache_dir = cache_dir / "training"
    training_manifest_path = training_cache_dir / "manifest.json"
    training_manifest = (
        _read_cache_manifest(training_manifest_path, training_key) if args.resume else None
    )
    try:
        training = (
            load_cached_training_data(training_cache_dir)
            if training_manifest is not None
            else None
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        training = None
    if training is None:
        _log("Building sparse rating matrix")
        training = load_training_data(args.train_reviews)
        save_training_data(training_cache_dir, training)
        _write_cache_manifest(training_manifest_path, training_key)
    else:
        _log("Rating-matrix checkpoint is valid; skipping CSV aggregation")

    dev_users = read_user_ids(args.dev_users)
    test_users = read_user_ids(args.test_users)
    qrels = load_qrels(args.dev_qrels)
    dev_targets = dev_pair_ratings(args.dev_reviews, dev_users, training.catalogue)

    mf_key = {
        "version": CACHE_VERSION,
        "training": training_key,
        "shape": list(training.ratings.shape),
        "nnz": training.ratings.nnz,
        "factors": args.factors,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "seed": args.seed,
    }
    mf_cache_dir = cache_dir / "matrix_factorization"
    mf_manifest_path = mf_cache_dir / "manifest.json"
    mf_manifest = _read_cache_manifest(mf_manifest_path, mf_key) if args.resume else None
    try:
        mandatory_mf = load_matrix_factorization(mf_cache_dir) if mf_manifest else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        mandatory_mf = None
    if mandatory_mf is None:
        _log("Training mandatory biased matrix factorization")
        mandatory_mf = BiasedMatrixFactorization(
            factors=args.factors,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            seed=args.seed,
        ).fit(training.ratings)
        save_matrix_factorization(mf_cache_dir, mandatory_mf)
        _write_cache_manifest(mf_manifest_path, mf_key)
    else:
        _log("Matrix-factorization checkpoint is valid; skipping SGD training")

    tuned_mf = mandatory_mf
    mf_tuning_report: dict[str, Any] | None = None
    if args.tune_mf:
        from advanced_models import tune_matrix_factorization, validation_triplets

        tuning_key = {
            "version": ADVANCED_CACHE_VERSION,
            "training": training_key,
            "dev_reviews": _file_signature(args.dev_reviews),
            "factors": _parse_int_grid(args.mf_factors_grid, "--mf-factors-grid"),
            "learning_rates": _parse_float_grid(
                args.mf_learning_rate_grid, "--mf-learning-rate-grid", minimum=1e-12
            ),
            "regularizations": _parse_float_grid(
                args.mf_regularization_grid, "--mf-regularization-grid", minimum=0
            ),
            "epochs": _parse_int_grid(args.mf_epochs_grid, "--mf-epochs-grid"),
            "patience": args.early_stopping_patience,
            "seed": args.seed,
        }
        tuning_dir = cache_dir / "matrix_factorization_tuned"
        tuning_manifest_path = tuning_dir / "manifest.json"
        tuning_manifest = (
            _read_cache_manifest(tuning_manifest_path, tuning_key) if args.resume else None
        )
        try:
            if tuning_manifest is not None:
                tuned_mf = load_matrix_factorization(tuning_dir)
                mf_tuning_report = json.loads(
                    (tuning_dir / "mf_tuning.json").read_text(encoding="utf-8")
                )
            else:
                raise FileNotFoundError
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            _log("Tuning matrix factorization with dev-only early stopping")
            tuned_mf, mf_tuning_report = tune_matrix_factorization(
                training.ratings,
                validation_triplets(training, dev_targets),
                factors=tuning_key["factors"],
                learning_rates=tuning_key["learning_rates"],
                regularizations=tuning_key["regularizations"],
                epochs=tuning_key["epochs"],
                seed=args.seed,
                patience=args.early_stopping_patience,
            )
            save_matrix_factorization(tuning_dir, tuned_mf)
            _write_json(tuning_dir / "mf_tuning.json", mf_tuning_report)
            _write_cache_manifest(tuning_manifest_path, tuning_key)
        _write_json(args.outdir / "mf_tuning.json", mf_tuning_report)

    def load_or_build_profile_variant(
        name: str,
        *,
        key_extra: Mapping[str, Any],
        aspect_mapper: Any | None = None,
        sentiment_scorer: Any | None = None,
        advanced_sentiment: bool = True,
    ) -> Any:
        variant_dir = cache_dir / f"profiles_{name}"
        variant_key = {
            "version": ADVANCED_CACHE_VERSION,
            "train_reviews": training_signature,
            "model": "en_core_web_md",
            "model_version": model_version,
            "threshold": args.similarity_threshold,
            "confidence_weighted": args.confidence_weighted,
            "shrinkage": args.aspect_shrinkage,
            "max_text_chars": args.max_text_chars,
            **dict(key_extra),
        }
        variant_manifest_path = variant_dir / "manifest.json"
        variant_manifest = (
            _read_cache_manifest(variant_manifest_path, variant_key) if args.resume else None
        )
        try:
            variant = load_profile_matrices(variant_dir) if variant_manifest else None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            variant = None
        if variant is None:
            _log(f"Building advanced NLP profile variant: {name}")
            variant = build_aspect_profiles(
                args.train_reviews,
                nlp=nlp,
                threshold=args.similarity_threshold,
                batch_size=args.batch_size,
                n_process=args.n_process,
                evidence_path=args.outdir / f"aspect_evidence_{name}.jsonl",
                evidence_limit=args.evidence_limit,
                progress_every=args.progress_every,
                aspect_mapper=aspect_mapper,
                sentiment_scorer=sentiment_scorer,
                confidence_weighted=args.confidence_weighted,
                shrinkage=args.aspect_shrinkage,
                max_text_chars=args.max_text_chars,
                advanced_sentiment=advanced_sentiment,
            )
            save_profile_matrices(variant_dir, variant)
            _write_cache_manifest(variant_manifest_path, variant_key)
        else:
            _log(f"Advanced profile checkpoint is valid: {name}")
        return variant

    profile_variants: dict[str, Any] = {"mandatory_vader": profiles}
    if args.enable_advanced_vader:
        profile_variants["advanced_vader"] = load_or_build_profile_variant(
            "advanced_vader",
            key_extra={"mapper": "spacy", "sentiment": "advanced_vader"},
        )

    if args.enable_word2vec:
        from advanced_nlp import Word2VecAspectMapper

        w2v_dir = cache_dir / "word2vec"
        w2v_key = {
            "version": ADVANCED_CACHE_VERSION,
            "train_reviews": training_signature,
            "vector_size": args.word2vec_vector_size,
            "window": args.word2vec_window,
            "min_count": args.word2vec_min_count,
            "epochs": args.word2vec_epochs,
            "workers": args.word2vec_workers,
            "seed": args.seed,
        }
        w2v_manifest_path = w2v_dir / "manifest.json"
        w2v_manifest = _read_cache_manifest(w2v_manifest_path, w2v_key) if args.resume else None
        try:
            mapper = Word2VecAspectMapper.load(w2v_dir / "model.bin") if w2v_manifest else None
        except (OSError, ValueError, RuntimeError):
            mapper = None
        if mapper is None:
            _log("Training train-only Word2Vec aspect mapper")
            mapper = Word2VecAspectMapper.train(
                args.train_reviews,
                vector_size=args.word2vec_vector_size,
                window=args.word2vec_window,
                min_count=args.word2vec_min_count,
                epochs=args.word2vec_epochs,
                seed=args.seed,
                workers=args.word2vec_workers,
            )
            mapper.save(w2v_dir / "model.bin")
            _write_cache_manifest(w2v_manifest_path, w2v_key)
        profile_variants["word2vec_vader"] = load_or_build_profile_variant(
            "word2vec_vader",
            key_extra={"mapper": w2v_key, "sentiment": "advanced_vader"},
            aspect_mapper=mapper,
        )

    if args.use_transformer:
        from advanced_nlp import TransformerABSAScorer

        transformer_key_extra = {
            "mapper": "spacy",
            "sentiment": args.transformer_model,
            "max_length": args.transformer_max_length,
        }
        transformer_key = {
            "version": ADVANCED_CACHE_VERSION,
            "train_reviews": training_signature,
            "model": "en_core_web_md",
            "model_version": model_version,
            "threshold": args.similarity_threshold,
            "confidence_weighted": args.confidence_weighted,
            "shrinkage": args.aspect_shrinkage,
            "max_text_chars": args.max_text_chars,
            **transformer_key_extra,
        }
        transformer_dir = cache_dir / "profiles_transformer"
        transformer_manifest = (
            _read_cache_manifest(transformer_dir / "manifest.json", transformer_key)
            if args.resume
            else None
        )
        try:
            transformer_profiles = (
                load_profile_matrices(transformer_dir)
                if transformer_manifest is not None
                else None
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            transformer_profiles = None
        if transformer_profiles is None:
            transformer = TransformerABSAScorer(
                args.transformer_model,
                device=args.transformer_device,
                batch_size=args.transformer_batch_size,
                max_length=args.transformer_max_length,
            )
            transformer_profiles = load_or_build_profile_variant(
                "transformer",
                key_extra=transformer_key_extra,
                sentiment_scorer=transformer,
                advanced_sentiment=False,
            )
        else:
            _log("Advanced profile checkpoint is valid: transformer")
        profile_variants["transformer"] = transformer_profiles

    bpr_model = None
    bpr_report: dict[str, Any] | None = None
    if args.train_bpr:
        from advanced_models import BayesianPersonalizedRanking, load_bpr, save_bpr

        bpr_key = {
            "version": ADVANCED_CACHE_VERSION,
            "training": training_key,
            "factors": args.bpr_factors,
            "epochs": args.bpr_epochs,
            "learning_rate": args.bpr_learning_rate,
            "regularization": args.bpr_regularization,
            "batch_size": args.bpr_batch_size,
            "seed": args.seed,
        }
        bpr_dir = cache_dir / "bpr"
        bpr_manifest_path = bpr_dir / "manifest.json"
        bpr_manifest = _read_cache_manifest(bpr_manifest_path, bpr_key) if args.resume else None
        try:
            bpr_model = load_bpr(bpr_dir) if bpr_manifest else None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            bpr_model = None
        if bpr_model is None:
            _log("Training BPR-MF on implicit train interactions")
            bpr_model = BayesianPersonalizedRanking(
                factors=args.bpr_factors,
                epochs=args.bpr_epochs,
                learning_rate=args.bpr_learning_rate,
                regularization=args.bpr_regularization,
                batch_size=args.bpr_batch_size,
                seed=args.seed,
            ).fit(training.ratings)
            save_bpr(bpr_dir, bpr_model)
            _write_cache_manifest(bpr_manifest_path, bpr_key)
        bpr_report = {"hyperparameters": bpr_key, "loss_history": bpr_model.loss_history}
        _write_json(args.outdir / "bpr_training.json", bpr_report)

    mandatory_recommender = HybridRecommender(training, mandatory_mf, profiles)
    primary_profiles = profile_variants.get("advanced_vader", profiles)
    primary_recommender = HybridRecommender(
        training, tuned_mf, primary_profiles, bpr_model=bpr_model
    )

    metrics: dict[str, dict[str, float]] = {}
    dev_rows_by_model: dict[str, list[dict[str, object]]] = {}
    model_registry: dict[str, tuple[HybridRecommender, dict[str, object]]] = {}

    def register_model(
        name: str,
        recommender: HybridRecommender,
        *,
        alpha: float,
        minimum: int,
        threshold: float,
        mode: str = "hybrid",
        popularity_weight: float = 0.15,
    ) -> dict[str, float]:
        kwargs: dict[str, object] = {
            "alpha": alpha,
            "top_k": args.top_k,
            "min_interactions": minimum,
            "threshold": threshold,
            "mode": mode,
            "apply_abstention": True,
            "popularity_weight": popularity_weight,
            "adaptive_cf_k": args.adaptive_cf_k,
            "adaptive_aspect_k": args.adaptive_aspect_k,
        }
        rows, _ = recommender.recommend_many(dev_users, **kwargs)
        if mode == "popularity":
            predictor = lambda _user, item: training.item_mean_ratings.get(
                item, training.global_mean
            )
        else:
            predictor = lambda user, item: 1.0 + 4.0 * recommender.score_mode(
                user,
                item,
                alpha=alpha,
                mode=mode,
                popularity_weight=popularity_weight,
                adaptive_cf_k=args.adaptive_cf_k,
                adaptive_aspect_k=args.adaptive_aspect_k,
            )
        value = evaluate(
            rows,
            qrels,
            dev_users,
            training.catalogue,
            dev_targets,
            predictor,
            k=args.top_k,
            seen_by_user=training.seen_by_user,
        )
        metrics[name] = value
        dev_rows_by_model[name] = rows
        model_registry[name] = (recommender, kwargs)
        return value

    register_model(
        "Aspect Content-Only", mandatory_recommender,
        alpha=0.0, minimum=args.min_interactions, threshold=args.threshold,
    )
    register_model(
        "Popularity-Based", mandatory_recommender,
        alpha=args.alpha, minimum=args.min_interactions, threshold=args.threshold,
        mode="popularity",
    )
    register_model(
        "Ratings-Only SVD", mandatory_recommender,
        alpha=1.0, minimum=args.min_interactions, threshold=args.threshold,
    )
    register_model(
        "VADER-Hybrid", mandatory_recommender,
        alpha=args.alpha, minimum=args.min_interactions, threshold=args.threshold,
    )

    sweep_recommender = primary_recommender
    if args.sweep:
        sweep_alphas = alpha_grid
        sweep_minimums = interaction_grid
        sweep_thresholds = threshold_grid
    else:
        sweep_alphas = sorted({0.0, 0.25, 0.5, 0.7, 1.0, args.alpha})
        sweep_minimums = [args.min_interactions]
        sweep_thresholds = [args.threshold]
    sweep_configurations: dict[str, dict[str, object]] = {}
    for alpha in sweep_alphas:
        for minimum in sweep_minimums:
            for threshold in sweep_thresholds:
                name = f"alpha={alpha:.10g}|M={minimum}|tau={threshold:.10g}"
                sweep_configurations[name] = {
                    "alpha": alpha,
                    "top_k": args.top_k,
                    "min_interactions": minimum,
                    "threshold": threshold,
                    "mode": "hybrid",
                    "apply_abstention": True,
                }
    _log(f"Vectorized hyperparameter sweep: {len(sweep_configurations)} settings")
    sweep_rows, _ = sweep_recommender.recommend_many_configurations(
        dev_users, sweep_configurations
    )
    sweep_records: list[dict[str, Any]] = []
    sweep_metrics: dict[str, dict[str, float]] = {}
    for name, config in sweep_configurations.items():
        alpha = float(config["alpha"])
        value = evaluate(
            sweep_rows[name],
            qrels,
            dev_users,
            training.catalogue,
            dev_targets,
            _star_predictor(sweep_recommender, alpha),
            k=args.top_k,
            seen_by_user=training.seen_by_user,
        )
        sweep_metrics[name] = value
        sweep_records.append({"configuration": name, **config, **value})
    best_sweep_name = min(
        sweep_metrics,
        key=lambda name: _metric_selection_key(name, sweep_metrics[name]),
    )
    best_sweep_config = sweep_configurations[best_sweep_name]
    register_model(
        "Dev-Tuned Hybrid",
        sweep_recommender,
        alpha=float(best_sweep_config["alpha"]),
        minimum=int(best_sweep_config["min_interactions"]),
        threshold=float(best_sweep_config["threshold"]),
    )

    tuned_alpha = float(best_sweep_config["alpha"])
    tuned_minimum = int(best_sweep_config["min_interactions"])
    tuned_threshold = float(best_sweep_config["threshold"])
    if args.tune_mf:
        register_model(
            "Ratings-Only SVD Tuned", primary_recommender,
            alpha=1.0, minimum=tuned_minimum, threshold=tuned_threshold,
        )
    if "advanced_vader" in profile_variants:
        register_model(
            "VADER-Hybrid+", primary_recommender,
            alpha=tuned_alpha, minimum=tuned_minimum, threshold=tuned_threshold,
        )
    if args.enable_advanced_rankers:
        register_model(
            "Popularity-Aware Hybrid+", primary_recommender,
            alpha=tuned_alpha, minimum=tuned_minimum, threshold=tuned_threshold,
            mode="popularity_hybrid", popularity_weight=args.popularity_weight,
        )
        register_model(
            "Adaptive Hybrid+", primary_recommender,
            alpha=tuned_alpha, minimum=tuned_minimum, threshold=tuned_threshold,
            mode="adaptive", popularity_weight=args.popularity_weight,
        )
    if bpr_model is not None:
        register_model(
            "BPR-MF", primary_recommender,
            alpha=tuned_alpha, minimum=tuned_minimum, threshold=tuned_threshold,
            mode="bpr",
        )
    if "word2vec_vader" in profile_variants:
        word2vec_recommender = HybridRecommender(
            training, tuned_mf, profile_variants["word2vec_vader"]
        )
        register_model(
            "Word2Vec VADER-Hybrid+", word2vec_recommender,
            alpha=tuned_alpha, minimum=tuned_minimum, threshold=tuned_threshold,
        )
    if "transformer" in profile_variants:
        transformer_recommender = HybridRecommender(
            training, tuned_mf, profile_variants["transformer"]
        )
        register_model(
            "Transformer ABSA-Hybrid+", transformer_recommender,
            alpha=tuned_alpha, minimum=tuned_minimum, threshold=tuned_threshold,
        )

    selected_model = "VADER-Hybrid"
    if args.auto_select:
        selected_model = min(
            metrics,
            key=lambda name: _metric_selection_key(name, metrics[name]),
        )
    selected_recommender, selected_kwargs = model_registry[selected_model]
    write_recommendations(
        args.outdir / "dev_recommendations.csv", dev_rows_by_model[selected_model]
    )
    _write_json(args.outdir / "dev_metrics.json", metrics)

    mandatory_test_kwargs = {
        "alpha": args.alpha,
        "top_k": args.top_k,
        "min_interactions": args.min_interactions,
        "threshold": args.threshold,
        "mode": "hybrid",
        "apply_abstention": True,
    }
    mandatory_test_rows, mandatory_test_results = mandatory_recommender.recommend_many(
        test_users, **mandatory_test_kwargs
    )
    selected_test_rows, selected_test_results = selected_recommender.recommend_many(
        test_users, **selected_kwargs
    )
    write_recommendations(
        args.outdir / "test_recommendations_mandatory.csv", mandatory_test_rows
    )
    write_recommendations(args.outdir / "test_recommendations.csv", selected_test_rows)

    ablation = {
        "selection_order": ["ndcg@10", "hr@10", "rmse"],
        "selected_model": selected_model,
        "best_sweep_configuration": best_sweep_name,
        "best_sweep_metrics": sweep_metrics[best_sweep_name],
        "sweep": sorted(
            sweep_records,
            key=lambda row: _metric_selection_key(
                str(row["configuration"]),
                {
                    "ndcg@10": float(row["ndcg@10"]),
                    "hr@10": float(row["hr@10"]),
                    "rmse": float(row["rmse"]),
                },
            ),
        ),
        "models": metrics,
    }
    _write_json(args.outdir / "ablation.json", ablation)
    _write_json(
        args.outdir / "aspect_diagnostics.json",
        {name: variant.stats for name, variant in profile_variants.items()},
    )

    names = _business_names(args.businesses)
    with (args.outdir / "trace.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as trace_file:
        for user_id in sorted(test_users)[:20]:
            trace_file.write(
                json.dumps(
                    _trace_entry(
                        mandatory_recommender,
                        user_id,
                        mandatory_test_results[user_id],
                        names,
                        args.alpha,
                        args.threshold,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (args.outdir / "trace_best.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as trace_file:
        for user_id in sorted(test_users)[:20]:
            result = selected_test_results[user_id]
            trace_file.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "selected_model": selected_model,
                        "decision_reason": result.reason,
                        "fallback": result.fallback,
                        "top_recommendation": result.rows[0] if result.rows else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    selected_rows = dev_rows_by_model[selected_model]
    selected_predictor_kwargs = model_registry[selected_model][1]
    selected_mode = str(selected_predictor_kwargs.get("mode", "hybrid"))
    selected_alpha = float(selected_predictor_kwargs.get("alpha", args.alpha))
    selected_pop_weight = float(
        selected_predictor_kwargs.get("popularity_weight", args.popularity_weight)
    )
    if selected_mode == "popularity":
        selected_predictor = lambda _user, item: training.item_mean_ratings.get(
            item, training.global_mean
        )
    else:
        selected_predictor = lambda user, item: 1.0 + 4.0 * selected_recommender.score_mode(
            user,
            item,
            alpha=selected_alpha,
            mode=selected_mode,
            popularity_weight=selected_pop_weight,
            adaptive_cf_k=args.adaptive_cf_k,
            adaptive_aspect_k=args.adaptive_aspect_k,
        )
    error_analysis: dict[str, Any] = {}
    for bucket, bucket_users in {
        "cold_lt_5": [u for u in dev_users if training.user_review_counts.get(u, 0) < 5],
        "warm_5_to_19": [u for u in dev_users if 5 <= training.user_review_counts.get(u, 0) < 20],
        "heavy_ge_20": [u for u in dev_users if training.user_review_counts.get(u, 0) >= 20],
    }.items():
        bucket_rows = [row for row in selected_rows if row["user_id"] in set(bucket_users)]
        error_analysis[bucket] = {
            "users": len(bucket_users),
            "metrics": (
                evaluate(
                    bucket_rows,
                    qrels,
                    bucket_users,
                    training.catalogue,
                    dev_targets,
                    selected_predictor,
                    k=args.top_k,
                    seen_by_user=training.seen_by_user,
                )
                if bucket_users
                else None
            ),
        }
    _write_json(args.outdir / "error_analysis.json", error_analysis)

    summary: dict[str, Any] = {
        "phase0": phase0_stats,
        "training_rows": training.row_count,
        "dev_rows": _count_csv_rows(args.dev_reviews),
        "dev_evaluation_pairs": len(dev_targets),
        "training_users": len(training.user_to_index),
        "training_catalogue_items": len(training.item_to_index),
        "selected_model": selected_model,
        "selected_dev_metrics": metrics[selected_model],
        "mandatory_nlp": profiles.stats,
        "profile_variants": sorted(profile_variants),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "versions": {
            "python": platform.python_version(),
            "spacy": _package_version("spacy"),
            "en_core_web_md": model_version,
            "nltk": _package_version("nltk"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "gensim": _package_version("gensim"),
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
        },
        "hyperparameters": {
            "canonical_aspects": list(CANONICAL_ASPECTS),
            "similarity_threshold": args.similarity_threshold,
            "batch_size": args.batch_size,
            "n_process": args.n_process,
            "mandatory": {
                "alpha": args.alpha,
                "top_k": args.top_k,
                "min_interactions": args.min_interactions,
                "abstention_threshold": args.threshold,
            },
            "selected_sweep": best_sweep_config,
            "confidence_weighted": args.confidence_weighted,
            "aspect_shrinkage": args.aspect_shrinkage,
            "matrix_factorization_fit_split": "train",
            "seed": args.seed,
        },
        "mf_tuning": mf_tuning_report,
        "bpr": bpr_report,
    }
    _write_json(args.outdir / "summary.json", summary)
    return {"metrics": metrics, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    result = run(args)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
