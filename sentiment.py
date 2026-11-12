from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass"})
OBJECT_DEPS = frozenset({"dobj", "obj", "pobj"})
OPINION_POS = frozenset({"ADJ", "VERB", "NOUN"})
CONTRAST_WORDS = frozenset({"but", "however", "although", "though", "yet"})


@dataclass(frozen=True)
class OpinionEvidence:
    """One dependency-linked opinion phrase and its structural confidence."""

    phrase: str
    relation: str
    confidence: float


@dataclass(frozen=True)
class AspectSentimentResult:
    """Detailed per-review aspect sentiment used by profile aggregation."""

    scores: dict[str, float]
    phrases: dict[str, list[str]]
    confidence: dict[str, float]
    evidence: dict[str, list[dict[str, object]]]


def load_vader_analyzer() -> Any:
    """Load the mandatory, unmodified NLTK VADER analyzer."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "NLTK is required. Install the pinned requirements before running."
        ) from exc
    try:
        return SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise RuntimeError(
            "The NLTK vader_lexicon resource is missing. Run: "
            "python -m nltk.downloader vader_lexicon"
        ) from exc


def _copular_predicate(aspect_head: Any) -> tuple[Any, str, float] | None:
    """Return an adjectival or nominal predicate linked to an aspect subject."""
    if getattr(aspect_head, "dep_", "") not in SUBJECT_DEPS:
        return None
    governor = getattr(aspect_head, "head", None)
    if governor is None:
        return None

    # spaCy normally heads "service is excellent/a disaster" at the predicate.
    if getattr(governor, "pos_", "") in {"ADJ", "NOUN"} and any(
        getattr(child, "dep_", "") == "cop" for child in governor.children
    ):
        relation = "copular_adjective" if governor.pos_ == "ADJ" else "copular_noun"
        return governor, relation, 1.0 if governor.pos_ == "ADJ" else 0.85

    # Also support verb-headed dependency representations.
    if (
        getattr(governor, "lemma_", "").lower() == "be"
        or getattr(governor, "dep_", "") == "cop"
    ):
        for child in governor.children:
            if (
                getattr(child, "pos_", "") in {"ADJ", "NOUN"}
                and getattr(child, "dep_", "") in {"acomp", "attr", "oprd"}
            ):
                relation = "copular_adjective" if child.pos_ == "ADJ" else "copular_noun"
                return child, relation, 1.0 if child.pos_ == "ADJ" else 0.85
    return None


def _copular_adjective(aspect_head: Any) -> Any | None:
    """Backward-compatible helper retained for the original assignment tests."""
    value = _copular_predicate(aspect_head)
    if value is None or getattr(value[0], "pos_", "") != "ADJ":
        return None
    return value[0]


def _true_negation(token: Any, siblings: Iterable[Any]) -> bool:
    """Exclude the idiom 'not only', whose *not* does not reverse polarity."""
    text = getattr(token, "text", "").lower()
    if text not in {"not", "n't", "never", "no"}:
        return True
    return not any(
        getattr(candidate, "text", "").lower() == "only"
        and getattr(candidate, "dep_", "") in {"advmod", "fixed"}
        for candidate in siblings
    )


def _opinion_phrase(opinion: Any) -> tuple[str, bool]:
    children = list(getattr(opinion, "children", ()))
    selected = [opinion]
    for child in children:
        dependency = getattr(child, "dep_", "")
        if dependency == "neg" and _true_negation(child, children):
            selected.append(child)
        elif dependency in {"advmod", "prt"}:
            selected.append(child)
        elif dependency == "cc" and getattr(child, "text", "").lower() in CONTRAST_WORDS:
            selected.append(child)
    selected.sort(key=lambda value: getattr(value, "i", 0))
    has_contrast = any(
        getattr(value, "text", "").lower() in CONTRAST_WORDS for value in selected
    )
    return " ".join(str(getattr(value, "text", "")) for value in selected), has_contrast


def _opinion_candidates(aspect_head: Any) -> list[tuple[Any, str, float]]:
    candidates: dict[int, tuple[Any, str, float]] = {}

    def add(token: Any, relation: str, confidence: float) -> None:
        if getattr(token, "pos_", "") not in OPINION_POS:
            return
        key = int(getattr(token, "i", id(token)))
        existing = candidates.get(key)
        if existing is None or confidence > existing[2]:
            candidates[key] = (token, relation, confidence)

    for child in getattr(aspect_head, "children", ()):
        dependency = getattr(child, "dep_", "")
        if dependency == "amod" and getattr(child, "pos_", "") == "ADJ":
            add(child, "amod", 1.0)
        elif dependency in {"acl", "relcl"} and getattr(child, "pos_", "") in {"ADJ", "VERB"}:
            add(child, "clausal_modifier", 0.8)

    copular = _copular_predicate(aspect_head)
    if copular is not None:
        add(*copular)

    governor = getattr(aspect_head, "head", None)
    if (
        governor is not None
        and getattr(governor, "pos_", "") == "VERB"
        and getattr(governor, "lemma_", "").lower() != "be"
        and getattr(aspect_head, "dep_", "") in SUBJECT_DEPS.union(OBJECT_DEPS)
    ):
        add(governor, "sentiment_verb", 0.9)

    # Coordinated opinions inherit the aspect link (fresh and delicious; good but slow).
    frontier = list(candidates.values())
    visited: set[int] = set()
    while frontier:
        token, _relation, parent_confidence = frontier.pop()
        key = int(getattr(token, "i", id(token)))
        if key in visited:
            continue
        visited.add(key)
        for child in getattr(token, "children", ()):
            if getattr(child, "dep_", "") != "conj":
                continue
            contrast = any(
                getattr(grandchild, "dep_", "") == "cc"
                and getattr(grandchild, "text", "").lower() in CONTRAST_WORDS
                for grandchild in getattr(child, "children", ())
            )
            confidence = min(parent_confidence, 0.9 if contrast else 0.85)
            relation = "contrast_conjunction" if contrast else "conjunction"
            add(child, relation, confidence)
            frontier.append((child, relation, confidence))

    return [candidates[index] for index in sorted(candidates)]


def extract_opinion_evidence(aspect_head: Any) -> list[OpinionEvidence]:
    """Extract local opinions, intensifiers, negation, verbs, nouns, and conjunctions."""
    evidence: list[OpinionEvidence] = []
    for token, relation, confidence in _opinion_candidates(aspect_head):
        phrase, has_contrast = _opinion_phrase(token)
        if has_contrast and relation != "contrast_conjunction":
            relation = "contrast"
            confidence *= 0.95
        if phrase:
            evidence.append(OpinionEvidence(phrase, relation, confidence))
    return evidence


def extract_opinion_phrases(aspect_head: Any) -> list[str]:
    """Return dependency-linked phrases while preserving the original public API."""
    return [value.phrase for value in extract_opinion_evidence(aspect_head)]


def _mandatory_opinion_phrases(aspect_head: Any) -> list[str]:
    """Assignment-exact amod/copular-adjective phrases.

    The advanced extractor above is kept as an opt-in experiment.  This helper
    preserves the mandatory baseline, which permits only adjectival modifiers
    and adjectival copular complements, with local advmod/neg tokens.
    """
    adjectives: dict[int, Any] = {}
    for child in getattr(aspect_head, "children", ()):
        if (
            getattr(child, "dep_", "") == "amod"
            and getattr(child, "pos_", "") == "ADJ"
        ):
            adjectives[int(getattr(child, "i", id(child)))] = child
    copular = _copular_adjective(aspect_head)
    if copular is not None:
        adjectives[int(getattr(copular, "i", id(copular)))] = copular

    phrases: list[str] = []
    for index in sorted(adjectives):
        adjective = adjectives[index]
        selected = [adjective]
        selected.extend(
            child
            for child in getattr(adjective, "children", ())
            if getattr(child, "dep_", "") in {"advmod", "neg"}
        )
        selected.sort(key=lambda token: int(getattr(token, "i", 0)))
        phrases.append(" ".join(str(getattr(token, "text", "")) for token in selected))
    return phrases


def score_review_aspects_mandatory_detailed(
    mentions: Iterable[tuple[str, Any]],
    analyzer: Any,
) -> AspectSentimentResult:
    """Assignment-exact, unweighted mean VADER baseline."""
    mentioned: set[str] = set()
    phrases_by_aspect: dict[str, list[str]] = defaultdict(list)
    for aspect, head in mentions:
        mentioned.add(aspect)
        phrases_by_aspect[aspect].extend(_mandatory_opinion_phrases(head))

    scores: dict[str, float] = {}
    confidence: dict[str, float] = {}
    details: dict[str, list[dict[str, object]]] = {}
    for aspect in sorted(mentioned):
        phrases = phrases_by_aspect.get(aspect, [])
        compounds = [
            float(analyzer.polarity_scores(phrase)["compound"])
            for phrase in phrases
        ]
        scores[aspect] = sum(compounds) / len(compounds) if compounds else 0.0
        confidence[aspect] = 1.0 if compounds else 0.0
        details[aspect] = [
            {
                "phrase": phrase,
                "relation": "mandatory_amod_or_copular_adjective",
                "compound": compound,
                "weight": 1.0,
            }
            for phrase, compound in zip(phrases, compounds)
        ]
    return AspectSentimentResult(
        scores,
        {key: value for key, value in sorted(phrases_by_aspect.items())},
        confidence,
        details,
    )


def score_review_aspects_detailed(
    mentions: Iterable[tuple[str, Any]],
    analyzer: Any,
) -> AspectSentimentResult:
    """Confidence-weighted VADER sentiment for every mentioned aspect."""
    mentioned: set[str] = set()
    evidence_by_aspect: dict[str, list[OpinionEvidence]] = defaultdict(list)
    for aspect, head in mentions:
        mentioned.add(aspect)
        evidence_by_aspect[aspect].extend(extract_opinion_evidence(head))

    scores: dict[str, float] = {}
    phrases: dict[str, list[str]] = {}
    confidence: dict[str, float] = {}
    details: dict[str, list[dict[str, object]]] = {}
    for aspect in sorted(mentioned):
        values = evidence_by_aspect.get(aspect, [])
        phrases[aspect] = [value.phrase for value in values]
        if not values:
            scores[aspect] = 0.0
            confidence[aspect] = 0.0
            details[aspect] = []
            continue

        weighted_sum = 0.0
        weight_sum = 0.0
        aspect_details: list[dict[str, object]] = []
        for value in values:
            compound = float(analyzer.polarity_scores(value.phrase)["compound"])
            # Weak lexical scores receive less influence, but never disappear entirely.
            weight = value.confidence * max(abs(compound), 0.15)
            weighted_sum += weight * compound
            weight_sum += weight
            aspect_details.append(
                {
                    "phrase": value.phrase,
                    "relation": value.relation,
                    "compound": compound,
                    "weight": weight,
                }
            )
        scores[aspect] = weighted_sum / weight_sum if weight_sum else 0.0
        confidence[aspect] = min(1.0, weight_sum / len(values))
        details[aspect] = aspect_details
    return AspectSentimentResult(scores, phrases, confidence, details)


def score_review_aspects(
    mentions: Iterable[tuple[str, Any]],
    analyzer: Any,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Backward-compatible wrapper for the mandatory assignment baseline."""
    result = score_review_aspects_mandatory_detailed(mentions, analyzer)
    return result.scores, result.phrases
