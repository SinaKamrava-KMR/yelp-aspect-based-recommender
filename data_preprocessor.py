from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Iterator


BUSINESS_OUTPUT_FIELDS = (
    "business_id",
    "name",
    "categories",
    "city",
    "stars",
    "review_count",
)

REVIEW_OUTPUT_FIELDS = (
    "review_id",
    "user_id",
    "business_id",
    "stars",
    "date",
    "text",
)

DEFAULT_CATEGORY_LABELS = frozenset({"Restaurants", "Food"})


def _iter_json_objects(
    input_path: Path,
    stats: Counter[str],
    namespace: str,
) -> Iterator[dict[str, Any]]:
    """Stream JSON Lines and count malformed/non-object records."""
    with input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stats[f"{namespace}_lines_total"] += 1
            if not line.strip():
                stats[f"{namespace}_blank_lines"] += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats[f"{namespace}_malformed_json"] += 1
                continue
            if not isinstance(record, dict):
                stats[f"{namespace}_non_object_json"] += 1
                continue
            yield record


def _business_discard_reason(record: dict[str, Any]) -> str | None:
    missing = [field for field in BUSINESS_OUTPUT_FIELDS if field not in record]
    if missing:
        return "missing_required"

    # The assignment explicitly treats a null category as no categories.
    for field in ("business_id", "name", "city"):
        value = record[field]
        if value is None:
            return "null_required"
        if not isinstance(value, str) or not value.strip():
            return "invalid_text_field"

    categories = record["categories"]
    if categories is not None and not isinstance(categories, str):
        return "invalid_text_field"

    stars = record["stars"]
    if stars is None:
        return "null_required"
    if isinstance(stars, bool) or not isinstance(stars, (int, float)):
        return "invalid_stars"
    numeric_stars = float(stars)
    if not math.isfinite(numeric_stars) or not 0.0 <= numeric_stars <= 5.0:
        return "invalid_stars"

    review_count = record["review_count"]
    if review_count is None:
        return "null_required"
    if (
        isinstance(review_count, bool)
        or not isinstance(review_count, int)
        or review_count < 0
    ):
        return "invalid_review_count"
    return None


def _has_allowed_category(
    categories: str | None,
    allowed_labels: frozenset[str],
) -> bool:
    if categories is None:
        return False
    tokens = {part.strip() for part in categories.split(",") if part.strip()}
    return bool(tokens.intersection(allowed_labels))


def preprocess_businesses(
    input_path: str | Path,
    output_path: str | Path,
    *,
    city: str = "Philadelphia",
    category_labels: frozenset[str] = DEFAULT_CATEGORY_LABELS,
) -> tuple[set[str], dict[str, int]]:
    """Filter, validate, de-duplicate, and sort Yelp businesses."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Business input file does not exist: {input_path}")
    if not isinstance(city, str) or not city:
        raise ValueError("city must be a non-empty string")
    if not category_labels:
        raise ValueError("category_labels must not be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    stats: Counter[str] = Counter()
    retained: dict[str, dict[str, Any]] = {}

    for record in _iter_json_objects(input_path, stats, "business"):
        reason = _business_discard_reason(record)
        if reason is not None:
            stats[f"business_discarded_{reason}"] += 1
            continue
        if record["city"] != city:
            stats["business_filtered_city"] += 1
            continue
        if not _has_allowed_category(record["categories"], category_labels):
            stats["business_filtered_category"] += 1
            continue
        business_id = record["business_id"]
        if business_id in retained:
            stats["business_duplicate_id"] += 1
            continue
        retained[business_id] = {
            field: (float(record[field]) if field == "stars" else record[field])
            for field in BUSINESS_OUTPUT_FIELDS
        }

    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for business_id in sorted(retained):
            output_file.write(
                json.dumps(retained[business_id], ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            stats["businesses_written"] += 1
    temporary_path.replace(output_path)
    return set(retained), dict(sorted(stats.items()))


def _normalize_iso_calendar_date(value: Any) -> str | None:
    if not isinstance(value, str) or value.strip() != value:
        return None
    if len(value) < 10 or value[4] != "-" or value[7] != "-":
        return None
    try:
        parsed = date.fromisoformat(value[:10])
        if len(value) > 10:
            datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.isoformat()


def _validate_review_record(
    record: dict[str, Any],
) -> tuple[str | None, str | None, float | None]:
    missing = [field for field in REVIEW_OUTPUT_FIELDS if field not in record]
    if missing:
        return "missing_required", None, None
    if any(record[field] is None for field in REVIEW_OUTPUT_FIELDS):
        return "null_required", None, None
    for field in ("review_id", "user_id", "business_id"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            return "invalid_text_field", None, None
    if not isinstance(record["text"], str):
        return "invalid_text_field", None, None
    stars = record["stars"]
    if isinstance(stars, bool) or not isinstance(stars, (int, float)):
        return "invalid_stars", None, None
    numeric_stars = float(stars)
    if not math.isfinite(numeric_stars) or not 1.0 <= numeric_stars <= 5.0:
        return "invalid_stars", None, None
    normalized_date = _normalize_iso_calendar_date(record["date"])
    if normalized_date is None:
        return "invalid_date", None, None
    return None, normalized_date, numeric_stars


def preprocess_reviews(
    input_path: str | Path,
    train_output_path: str | Path,
    dev_output_path: str | Path,
    *,
    eligible_business_ids: Iterable[str],
    split_date: str = "2021-01-01",
    commit_interval: int = 10_000,
    progress_every: int = 1_000_000,
) -> dict[str, int]:
    """Stream reviews through SQLite, then deterministically split sorted rows."""
    input_path = Path(input_path)
    train_output_path = Path(train_output_path)
    dev_output_path = Path(dev_output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Review input file does not exist: {input_path}")
    if _normalize_iso_calendar_date(split_date) != split_date or len(split_date) != 10:
        raise ValueError("split_date must use the YYYY-MM-DD format")
    if isinstance(commit_interval, bool) or commit_interval < 1:
        raise ValueError("commit_interval must be a positive integer")
    if isinstance(progress_every, bool) or progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if train_output_path.resolve() == dev_output_path.resolve():
        raise ValueError("Train and dev output paths must be different")

    allowed_ids = set(eligible_business_ids)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    dev_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_tmp = train_output_path.with_suffix(train_output_path.suffix + ".tmp")
    dev_tmp = dev_output_path.with_suffix(dev_output_path.suffix + ".tmp")
    stats: Counter[str] = Counter()

    with TemporaryDirectory(prefix=".review_sort_", dir=train_output_path.parent) as tmpdir:
        connection = sqlite3.connect(Path(tmpdir) / "reviews.sqlite3")
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(
                """
                CREATE TABLE reviews (
                    review_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    business_id TEXT NOT NULL,
                    stars REAL NOT NULL,
                    review_date TEXT NOT NULL,
                    text TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            pending = 0
            next_progress = progress_every
            for record in _iter_json_objects(input_path, stats, "review"):
                if progress_every and stats["review_lines_total"] >= next_progress:
                    print(
                        "[Phase 0] scanned "
                        f"{stats['review_lines_total']:,} raw review lines",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_progress += progress_every
                reason, review_date, numeric_stars = _validate_review_record(record)
                if reason is not None:
                    stats[f"review_discarded_{reason}"] += 1
                    continue
                if record["business_id"] not in allowed_ids:
                    stats["review_filtered_business"] += 1
                    continue
                assert review_date is not None and numeric_stars is not None
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO reviews
                    (review_id,user_id,business_id,stars,review_date,text)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        record["review_id"],
                        record["user_id"],
                        record["business_id"],
                        numeric_stars,
                        review_date,
                        record["text"],
                    ),
                )
                if cursor.rowcount == 0:
                    stats["review_duplicate_id"] += 1
                    # Keep the earlier date; equal-date ties retain first input row.
                    connection.execute(
                        """
                        UPDATE reviews
                        SET user_id=?,business_id=?,stars=?,review_date=?,text=?
                        WHERE review_id=? AND review_date>?
                        """,
                        (
                            record["user_id"],
                            record["business_id"],
                            numeric_stars,
                            review_date,
                            record["text"],
                            record["review_id"],
                            review_date,
                        ),
                    )
                pending += 1
                if pending >= commit_interval:
                    connection.commit()
                    pending = 0
            connection.commit()
            connection.execute(
                "CREATE INDEX reviews_sort_idx ON reviews(review_date,review_id)"
            )
            with (
                train_tmp.open("w", encoding="utf-8", newline="") as train_file,
                dev_tmp.open("w", encoding="utf-8", newline="") as dev_file,
            ):
                train_writer = csv.writer(train_file, lineterminator="\r\n")
                dev_writer = csv.writer(dev_file, lineterminator="\r\n")
                train_writer.writerow(REVIEW_OUTPUT_FIELDS)
                dev_writer.writerow(REVIEW_OUTPUT_FIELDS)
                rows = connection.execute(
                    """
                    SELECT review_id,user_id,business_id,stars,review_date,text
                    FROM reviews ORDER BY review_date ASC,review_id ASC
                    """
                )
                for row in rows:
                    writer = train_writer if row[4] < split_date else dev_writer
                    writer.writerow(row)
                    stats[
                        "reviews_written_train" if row[4] < split_date
                        else "reviews_written_dev"
                    ] += 1
        finally:
            connection.close()

    train_tmp.replace(train_output_path)
    dev_tmp.replace(dev_output_path)
    stats["reviews_written_total"] = (
        stats["reviews_written_train"] + stats["reviews_written_dev"]
    )
    return dict(sorted(stats.items()))


def _validated_csv_rows(path: Path, required_fields: tuple[str, ...]) -> Iterator[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != list(required_fields):
            raise ValueError(
                f"Invalid header in {path}: expected {list(required_fields)}, "
                f"got {reader.fieldnames}"
            )
        yield from reader


def generate_dev_artifacts(
    train_reviews_path: str | Path,
    dev_reviews_path: str | Path,
    dev_qrels_path: str | Path,
    dev_users_path: str | Path,
    *,
    min_train_rows: int = 3,
) -> dict[str, int]:
    """Generate collapsed dev qrels and eligible dev-user IDs."""
    if isinstance(min_train_rows, bool) or min_train_rows < 1:
        raise ValueError("min_train_rows must be a positive integer")
    train_path = Path(train_reviews_path)
    dev_path = Path(dev_reviews_path)
    qrels_path = Path(dev_qrels_path)
    users_path = Path(dev_users_path)
    if qrels_path.resolve() == users_path.resolve():
        raise ValueError("dev_qrels and dev_users paths must differ")

    train_user_counts: Counter[str] = Counter()
    catalogue: set[str] = set()
    for row in _validated_csv_rows(train_path, REVIEW_OUTPUT_FIELDS):
        train_user_counts[row["user_id"]] += 1
        catalogue.add(row["business_id"])

    pair_relevance: dict[tuple[str, str], int] = {}
    stats: Counter[str] = Counter()
    for row in _validated_csv_rows(dev_path, REVIEW_OUTPUT_FIELDS):
        stats["dev_rows_read"] += 1
        try:
            stars = float(row["stars"])
        except ValueError as exc:
            raise ValueError(f"Invalid dev stars value: {row['stars']!r}") from exc
        key = (row["user_id"], row["business_id"])
        pair_relevance[key] = max(pair_relevance.get(key, 0), int(stars >= 4.0))

    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    users_path.parent.mkdir(parents=True, exist_ok=True)
    qrels_tmp = qrels_path.with_suffix(qrels_path.suffix + ".tmp")
    users_tmp = users_path.with_suffix(users_path.suffix + ".tmp")
    with qrels_tmp.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(("user_id", "business_id", "relevance"))
        for (user_id, business_id), relevance in sorted(pair_relevance.items()):
            writer.writerow((user_id, business_id, relevance))
            stats["dev_qrels_rows"] += 1

    eligible_users = sorted(
        {
            user_id
            for (user_id, business_id), relevance in pair_relevance.items()
            if relevance == 1
            and business_id in catalogue
            and train_user_counts[user_id] >= min_train_rows
        }
    )
    with users_tmp.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\r\n")
        writer.writerow(("user_id",))
        writer.writerows((user_id,) for user_id in eligible_users)
    qrels_tmp.replace(qrels_path)
    users_tmp.replace(users_path)
    stats["dev_users_written"] = len(eligible_users)
    stats["training_catalogue_size"] = len(catalogue)
    return dict(sorted(stats.items()))


def find_raw_yelp_file(raw_dir: str | Path, filename: str) -> Path:
    """Find exactly one real Yelp JSON file, ignoring AppleDouble metadata."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)
    matches = sorted(
        path
        for path in raw_dir.rglob(filename)
        if path.is_file() and not path.name.startswith("._") and "__MACOSX" not in path.parts
    )
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {raw_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple copies of {filename} found: {matches}")
    return matches[0]


def run_phase0(
    raw_dir: str | Path,
    businesses_output: str | Path,
    train_output: str | Path,
    dev_output: str | Path,
    qrels_output: str | Path,
    dev_users_output: str | Path,
    *,
    city: str = "Philadelphia",
    category_labels: frozenset[str] = DEFAULT_CATEGORY_LABELS,
    split_date: str = "2021-01-01",
    min_train_rows: int = 3,
) -> dict[str, int]:
    """Regenerate every Phase 0 file except course-provided test_users.csv."""
    business_raw = find_raw_yelp_file(raw_dir, "yelp_academic_dataset_business.json")
    review_raw = find_raw_yelp_file(raw_dir, "yelp_academic_dataset_review.json")
    eligible_ids, business_stats = preprocess_businesses(
        business_raw,
        businesses_output,
        city=city,
        category_labels=category_labels,
    )
    review_stats = preprocess_reviews(
        review_raw,
        train_output,
        dev_output,
        eligible_business_ids=eligible_ids,
        split_date=split_date,
    )
    dev_stats = generate_dev_artifacts(
        train_output,
        dev_output,
        qrels_output,
        dev_users_output,
        min_train_rows=min_train_rows,
    )
    return dict(sorted({**business_stats, **review_stats, **dev_stats}.items()))
