import csv
import tempfile
import unittest
from pathlib import Path

from text_processor import (
    CANONICAL_ASPECTS,
    build_aspect_profiles,
    extract_aspect_mentions,
    load_profile_matrices,
    map_token_to_aspect,
    save_profile_matrices,
)


class Seed:
    def __init__(self, name):
        self.name = name
        self.has_vector = True
        self.vector_norm = 1.0


class Token:
    def __init__(self, text, scores, *, pos="NOUN", stop=False, vector=True):
        self.text = text
        self.lemma_ = text.lower()
        self.i = 0
        self.pos_ = pos
        self.is_stop = stop
        self.is_punct = False
        self.has_vector = vector
        self.vector_norm = 1.0 if vector else 0.0
        self.scores = scores
        self.children = []

    def similarity(self, seed):
        return self.scores[seed.name]


class Chunk:
    def __init__(self, root):
        self.root = root


class Doc:
    def __init__(self, tokens):
        self.noun_chunks = [Chunk(token) for token in tokens]

    def __len__(self):
        return len(self.noun_chunks) * 2


class FakeNLP:
    def __init__(self, docs):
        self.docs = docs
        self.pipe_call = None

    def make_doc(self, text):
        return [Seed(text)]

    def pipe(self, stream, *, as_tuples, batch_size, n_process):
        self.pipe_call = (as_tuples, batch_size, n_process)
        for text, context in stream:
            yield self.docs[text], context


class Analyzer:
    def polarity_scores(self, phrase):
        return {"compound": {"great": 0.6}[phrase]}


class TextProcessorTests(unittest.TestCase):
    def setUp(self):
        self.seeds = {aspect: Seed(aspect) for aspect in CANONICAL_ASPECTS}

    def test_threshold_and_fixed_tie_breaking(self):
        tied = {aspect: 0.5 for aspect in CANONICAL_ASPECTS}
        self.assertEqual(map_token_to_aspect(Token("meal", tied), self.seeds), ("food", 0.5))
        low = {aspect: 0.349 for aspect in CANONICAL_ASPECTS}
        self.assertIsNone(map_token_to_aspect(Token("thing", low), self.seeds))

    def test_only_eligible_noun_chunk_roots_are_emitted(self):
        food_scores = {aspect: (0.8 if aspect == "food" else 0.1) for aspect in CANONICAL_ASPECTS}
        mentions = extract_aspect_mentions(
            Doc([
                Token("pasta", food_scores),
                Token("it", food_scores, pos="PRON"),
                Token("the", food_scores, stop=True),
                Token("unknown", food_scores, vector=False),
            ]),
            self.seeds,
        )
        self.assertEqual([mention.aspect for mention in mentions], ["food"])
        self.assertEqual(mentions[0].root_text, "pasta")

    def test_profile_aggregation_and_required_pipe_arguments(self):
        food_scores = {aspect: (0.8 if aspect == "food" else 0.1) for aspect in CANONICAL_ASPECTS}
        service_scores = {aspect: (0.9 if aspect == "service" else 0.1) for aspect in CANONICAL_ASPECTS}
        pasta = Token("pasta", food_scores)
        adjective = Token("great", food_scores)
        adjective.pos_ = "ADJ"
        adjective.dep_ = "amod"
        adjective.i = 1
        pasta.children = [adjective]
        staff = Token("staff", service_scores)
        fake_nlp = FakeNLP({"doc-food": Doc([pasta]), "doc-service": Doc([staff])})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file, lineterminator="\r\n")
                writer.writerow(["review_id", "user_id", "business_id", "stars", "date", "text"])
                writer.writerow(["r1", "u1", "b1", 5, "2020-01-01", "doc-food"])
                writer.writerow(["r2", "u1", "b1", 4, "2020-01-02", "doc-service"])
            profiles = build_aspect_profiles(
                path,
                nlp=fake_nlp,
                analyzer=Analyzer(),
                batch_size=1000,
                n_process=3,
                evidence_path=Path(directory) / "evidence.jsonl",
                evidence_limit=1,
            )
            save_profile_matrices(Path(directory) / "profile-cache", profiles)
            cached = load_profile_matrices(Path(directory) / "profile-cache")
            evidence_lines = (
                Path(directory) / "evidence.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        self.assertEqual(fake_nlp.pipe_call, (True, 1000, 3))
        self.assertEqual(len(evidence_lines), 1)
        self.assertAlmostEqual(profiles.user_vector("u1")[0], 0.5)
        self.assertAlmostEqual(profiles.user_vector("u1")[1], 0.5)
        self.assertAlmostEqual(profiles.item_vector("b1")[0], 0.6)
        self.assertAlmostEqual(profiles.item_vector("b1")[1], 0.0)
        self.assertEqual(cached.user_to_index, profiles.user_to_index)
        self.assertEqual(cached.item_to_index, profiles.item_to_index)
        self.assertEqual(cached.stats, profiles.stats)


if __name__ == "__main__":
    unittest.main()
