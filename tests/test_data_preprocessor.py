import csv
import json
import tempfile
import unittest
from pathlib import Path

from data_preprocessor import (
    generate_dev_artifacts,
    preprocess_businesses,
    preprocess_reviews,
)


class DataPreprocessorTests(unittest.TestCase):
    def test_phase0_end_to_end_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_businesses = root / "business.json"
            businesses = root / "businesses.jsonl"
            raw_businesses.write_text(
                "\n".join(
                    [
                        json.dumps({
                            "business_id": "b2", "name": "Two",
                            "categories": "Italian, Restaurants",
                            "city": "Philadelphia", "stars": 4.0,
                            "review_count": 2,
                        }),
                        json.dumps({
                            "business_id": "b1", "name": "One",
                            "categories": "Food", "city": "Philadelphia",
                            "stars": 3.5, "review_count": 3,
                        }),
                        json.dumps({
                            "business_id": "wrong-city", "name": "Wrong",
                            "categories": "Food", "city": "Boston",
                            "stars": 4.0, "review_count": 1,
                        }),
                        json.dumps({
                            "business_id": "null-category", "name": "Null",
                            "categories": None, "city": "Philadelphia",
                            "stars": 4.0, "review_count": 1,
                        }),
                        json.dumps({
                            "business_id": "b1", "name": "Duplicate",
                            "categories": "Food", "city": "Philadelphia",
                            "stars": 1.0, "review_count": 1,
                        }),
                        '{"broken":',
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            eligible, business_stats = preprocess_businesses(raw_businesses, businesses)
            self.assertEqual(eligible, {"b1", "b2"})
            written = [json.loads(line) for line in businesses.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["business_id"] for row in written], ["b1", "b2"])
            self.assertEqual(business_stats["business_filtered_category"], 1)
            self.assertEqual(business_stats["business_duplicate_id"], 1)

            def review(review_id, user, item, stars, review_date, text="text"):
                return json.dumps({
                    "review_id": review_id, "user_id": user,
                    "business_id": item, "stars": stars,
                    "date": review_date, "text": text,
                })

            raw_reviews = root / "review.json"
            raw_reviews.write_text(
                "\n".join(
                    [
                        review("r3", "u1", "b1", 5, "2020-03-01"),
                        review("r1", "u1", "b2", 3, "2020-01-01"),
                        review("r2", "u1", "b1", 4, "2020-02-01"),
                        review("dup", "u1", "b1", 2, "2021-03-01"),
                        review("dup", "u1", "b2", 5, "2020-04-01"),
                        review("d1", "u1", "b2", 4, "2021-01-01"),
                        review("d2", "u2", "b1", 5, "2021-02-01"),
                        review("outside", "u1", "x", 5, "2020-01-01"),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            train = root / "train.csv"
            dev = root / "dev.csv"
            review_stats = preprocess_reviews(
                raw_reviews, train, dev, eligible_business_ids=eligible
            )
            with train.open("r", encoding="utf-8", newline="") as file:
                train_rows = list(csv.DictReader(file))
            self.assertEqual(
                [row["review_id"] for row in train_rows],
                ["r1", "r2", "r3", "dup"],
            )
            self.assertEqual(train_rows[-1]["business_id"], "b2")
            self.assertEqual(review_stats["review_duplicate_id"], 1)

            qrels = root / "dev_qrels.tsv"
            users = root / "dev_users.csv"
            dev_stats = generate_dev_artifacts(train, dev, qrels, users)
            self.assertEqual(dev_stats["dev_qrels_rows"], 2)
            self.assertEqual(users.read_text(encoding="utf-8").splitlines(), ["user_id", "u1"])
            with qrels.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file, delimiter="\t"))
            self.assertEqual(rows[0], {"user_id": "u1", "business_id": "b2", "relevance": "1"})
            self.assertEqual(rows[1], {"user_id": "u2", "business_id": "b1", "relevance": "1"})

    def test_dev_qrels_collapses_pair_with_max_relevance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = ["review_id", "user_id", "business_id", "stars", "date", "text"]
            train = root / "train.csv"
            dev = root / "dev.csv"
            with train.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file, lineterminator="\r\n")
                writer.writerow(header)
                for index in range(3):
                    writer.writerow([f"t{index}", "u", "b1", 3, "2020-01-01", "x"])
            with dev.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file, lineterminator="\r\n")
                writer.writerow(header)
                writer.writerow(["d1", "u", "b1", 2, "2021-01-01", "x"])
                writer.writerow(["d2", "u", "b1", 4, "2021-01-02", "x"])
            qrels, users = root / "qrels.tsv", root / "users.csv"
            generate_dev_artifacts(train, dev, qrels, users)
            self.assertIn("u\tb1\t1", qrels.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
