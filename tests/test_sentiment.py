import unittest

from sentiment import extract_opinion_phrases, score_review_aspects


class Token:
    def __init__(self, text, index, *, dep="", pos="", lemma=""):
        self.text = text
        self.i = index
        self.dep_ = dep
        self.pos_ = pos
        self.lemma_ = lemma or text.lower()
        self.children = []
        self.head = self

    def add(self, child):
        self.children.append(child)
        child.head = self
        return child


class Analyzer:
    def polarity_scores(self, phrase):
        values = {
            "extremely clean": 0.8,
            "not good": -0.5,
            "excellent": 0.7,
        }
        return {"compound": values[phrase]}


class SentimentTests(unittest.TestCase):
    def test_amod_modifiers_and_negation_are_local(self):
        service = Token("service", 2, pos="NOUN")
        clean = service.add(Token("clean", 4, dep="amod", pos="ADJ"))
        clean.add(Token("extremely", 3, dep="advmod", pos="ADV"))
        good = service.add(Token("good", 7, dep="amod", pos="ADJ"))
        good.add(Token("not", 6, dep="neg", pos="PART"))
        self.assertEqual(
            extract_opinion_phrases(service),
            ["extremely clean", "not good"],
        )

    def test_copular_adjective_and_review_mean(self):
        service = Token("service", 0, dep="nsubj", pos="NOUN")
        excellent = Token("excellent", 2, pos="ADJ")
        excellent.add(Token("is", 1, dep="cop", pos="AUX", lemma="be"))
        service.head = excellent
        scores, phrases = score_review_aspects(
            [("service", service), ("food", Token("food", 3, pos="NOUN"))],
            Analyzer(),
        )
        self.assertEqual(phrases["service"], ["excellent"])
        self.assertAlmostEqual(scores["service"], 0.7)
        self.assertEqual(scores["food"], 0.0)


if __name__ == "__main__":
    unittest.main()
