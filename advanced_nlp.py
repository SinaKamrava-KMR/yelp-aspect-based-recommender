from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from sentiment import AspectSentimentResult
from text_processor import CANONICAL_ASPECTS


class _ReviewCorpus:
    """Re-iterable, train-only corpus for Gensim's multi-pass Word2Vec trainer."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self) -> Iterator[list[str]]:
        try:
            from gensim.utils import simple_preprocess
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Word2Vec requires requirements-advanced.txt (gensim)."
            ) from exc
        with self.path.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            required = {"review_id", "user_id", "business_id", "stars", "date", "text"}
            if set(reader.fieldnames or ()) != required:
                raise ValueError(f"Unexpected training CSV header: {reader.fieldnames}")
            for row in reader:
                tokens = simple_preprocess(row["text"], deacc=False, min_len=2, max_len=30)
                if tokens:
                    yield tokens


class Word2VecAspectMapper:
    """Map noun roots to canonical aspects using train-fitted Word2Vec vectors."""

    name = "word2vec"

    def __init__(self, model: Any) -> None:
        self.model = model
        missing = [aspect for aspect in CANONICAL_ASPECTS if aspect not in model.wv]
        if missing:
            raise RuntimeError(
                "Word2Vec vocabulary is missing canonical seeds: " + ", ".join(missing)
            )

    @classmethod
    def train(
        cls,
        train_reviews_path: str | Path,
        *,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 3,
        epochs: int = 5,
        seed: int = 42,
        workers: int = 1,
    ) -> "Word2VecAspectMapper":
        try:
            from gensim.models import Word2Vec
            from gensim.utils import RULE_DEFAULT, RULE_KEEP
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Word2Vec requires requirements-advanced.txt (gensim)."
            ) from exc
        path = Path(train_reviews_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if vector_size < 10 or window < 1 or min_count < 1 or epochs < 1:
            raise ValueError("Invalid Word2Vec hyperparameters")

        canonical = frozenset(CANONICAL_ASPECTS)

        def keep_seeds(word: str, _count: int, _min_count: int) -> int:
            return RULE_KEEP if word in canonical else RULE_DEFAULT

        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1,
            negative=10,
            workers=workers,
            seed=seed,
            sorted_vocab=1,
        )
        corpus = _ReviewCorpus(path)
        model.build_vocab(corpus, trim_rule=keep_seeds)
        if model.corpus_count == 0:
            raise ValueError("Training reviews contain no Word2Vec tokens")
        model.train(corpus, total_examples=model.corpus_count, epochs=epochs)
        return cls(model)

    @classmethod
    def load(cls, path: str | Path) -> "Word2VecAspectMapper":
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Loading Word2Vec requires requirements-advanced.txt (gensim)."
            ) from exc
        return cls(Word2Vec.load(str(path)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def map_token(
        self,
        token: Any,
        *,
        threshold: float,
    ) -> tuple[str, float, float] | None:
        lemma = str(getattr(token, "lemma_", getattr(token, "text", ""))).lower()
        if lemma not in self.model.wv:
            return None
        values = [
            (aspect, float(self.model.wv.similarity(lemma, aspect)))
            for aspect in CANONICAL_ASPECTS
        ]
        ranked = sorted(
            enumerate(values),
            key=lambda value: (-value[1][1], value[0]),
        )
        _index, (aspect, score) = ranked[0]
        if not math.isfinite(score) or score < threshold:
            return None
        second = ranked[1][1][1] if len(ranked) > 1 else -1.0
        return aspect, score, score - second


class TransformerABSAScorer:
    """Optional Transformer polarity scorer for aspect-conditioned sentences."""

    name = "transformer"

    def __init__(
        self,
        model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
        *,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Transformer ABSA requires requirements-transformer.txt."
            ) from exc
        if batch_size < 1 or max_length < 32:
            raise ValueError("Invalid Transformer batch_size or max_length")
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.torch = torch
        self.device = torch.device(resolved_device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.negative_index, self.positive_index = self._label_indices()

    def _label_indices(self) -> tuple[int, int]:
        raw = getattr(self.model.config, "id2label", {}) or {}
        labels = {int(index): str(label).lower() for index, label in raw.items()}
        negative = next((index for index, label in labels.items() if "neg" in label), None)
        positive = next((index for index, label in labels.items() if "pos" in label), None)
        label_count = int(getattr(self.model.config, "num_labels", len(labels)))
        if negative is None or positive is None:
            if label_count != 3:
                raise RuntimeError(
                    "Transformer label mapping must expose positive/negative labels "
                    "or use the conventional three-class order."
                )
            negative, positive = 0, 2
        return negative, positive

    @staticmethod
    def _sentence_text(head: Any) -> str:
        try:
            value = str(head.sent.text).strip()
            if value:
                return value
        except (AttributeError, ValueError):
            pass
        try:
            value = str(head.doc.text).strip()
            if value:
                return value
        except AttributeError:
            pass
        return str(getattr(head, "text", ""))

    def _predict(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        results: list[tuple[float, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [value[0] for value in batch],
                [value[1] for value in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = self.torch.softmax(logits, dim=-1).cpu().numpy()
            for probability in probabilities:
                score = float(
                    probability[self.positive_index]
                    - probability[self.negative_index]
                )
                results.append((score, float(np.max(probability))))
        return results

    def score_review(self, mentions: Iterable[tuple[str, Any]]) -> AspectSentimentResult:
        materialized = list(mentions)
        if not materialized:
            return AspectSentimentResult({}, {}, {}, {})
        pairs = [(self._sentence_text(head), aspect) for aspect, head in materialized]
        predictions = self._predict(pairs)
        weighted_scores: dict[str, float] = defaultdict(float)
        weights: dict[str, float] = defaultdict(float)
        phrases: dict[str, list[str]] = defaultdict(list)
        details: dict[str, list[dict[str, object]]] = defaultdict(list)
        for (aspect, _head), (sentence, _target), (score, confidence) in zip(
            materialized, pairs, predictions
        ):
            weighted_scores[aspect] += confidence * score
            weights[aspect] += confidence
            phrases[aspect].append(sentence)
            details[aspect].append(
                {
                    "phrase": sentence,
                    "relation": "transformer_absa",
                    "compound": score,
                    "weight": confidence,
                }
            )
        scores = {
            aspect: weighted_scores[aspect] / weights[aspect]
            for aspect in sorted(weights)
        }
        confidence = {
            aspect: min(1.0, weights[aspect] / len(phrases[aspect]))
            for aspect in sorted(weights)
        }
        return AspectSentimentResult(
            scores,
            {aspect: phrases[aspect] for aspect in sorted(phrases)},
            confidence,
            {aspect: details[aspect] for aspect in sorted(details)},
        )
