import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

import system
from text_processor import ProfileMatrices


class FakeNLP:
    meta = {"version": "3.7.1"}


class SystemAdvancedTests(unittest.TestCase):
    def test_advanced_pipeline_writes_selected_and_mandatory_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            dev = root / "dev.csv"
            businesses = root / "businesses.jsonl"
            dev_users = root / "dev_users.csv"
            test_users = root / "test_users.csv"
            qrels = root / "qrels.tsv"
            out = root / "results"
            header = ["review_id", "user_id", "business_id", "stars", "date", "text"]
            train_rows = [
                ["r1", "u1", "b1", 5, "2020-01-01", "x"],
                ["r2", "u1", "b2", 4, "2020-01-02", "x"],
                ["r3", "u2", "b2", 3, "2020-01-03", "x"],
                ["r4", "u2", "b3", 2, "2020-01-04", "x"],
                ["r5", "u3", "b3", 5, "2020-01-05", "x"],
                ["r6", "u3", "b4", 4, "2020-01-06", "x"],
                ["r7", "u4", "b4", 3, "2020-01-07", "x"],
                ["r8", "u4", "b1", 2, "2020-01-08", "x"],
            ]
            with train.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file, lineterminator="\r\n")
                writer.writerow(header)
                writer.writerows(train_rows)
            with dev.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file, lineterminator="\r\n")
                writer.writerow(header)
                writer.writerows([
                    ["d1", "u1", "b3", 5, "2021-01-01", "x"],
                    ["d2", "u2", "b4", 4, "2021-01-02", "x"],
                ])
            businesses.write_text(
                "\n".join(
                    json.dumps({"business_id": item, "name": item.upper()})
                    for item in ("b1", "b2", "b3", "b4")
                ) + "\n",
                encoding="utf-8",
            )
            dev_users.write_text("user_id\r\nu1\r\nu2\r\n", encoding="utf-8")
            test_users.write_text("user_id\r\nu3\r\n", encoding="utf-8")
            qrels.write_text(
                "user_id\tbusiness_id\trelevance\r\n"
                "u1\tb3\t1\r\n"
                "u2\tb4\t1\r\n",
                encoding="utf-8",
            )

            users = {user: index for index, user in enumerate(("u1", "u2", "u3", "u4"))}
            items = {item: index for index, item in enumerate(("b1", "b2", "b3", "b4"))}
            fake_profiles = ProfileMatrices(
                csr_matrix(np.eye(4, 5)),
                csr_matrix(np.eye(4, 5)),
                users,
                items,
                {"diagnostics": {"review_aspect_coverage": 1.0}},
                {user: 2.0 for user in users},
                {item: 2.0 for item in items},
            )
            argv = [
                "--train-reviews", str(train),
                "--businesses", str(businesses),
                "--dev-reviews", str(dev),
                "--dev-qrels", str(qrels),
                "--dev-users", str(dev_users),
                "--test-users", str(test_users),
                "--outdir", str(out),
                "--enable-advanced-vader",
                "--enable-advanced-rankers",
                "--sweep",
                "--auto-select",
                "--tune-mf",
                "--train-bpr",
                "--alpha-grid", "0.5,1",
                "--min-interactions-grid", "1,2",
                "--threshold-grid", "0,0.3",
                "--mf-factors-grid", "2",
                "--mf-learning-rate-grid", "0.01",
                "--mf-regularization-grid", "0.05",
                "--mf-epochs-grid", "3",
                "--early-stopping-patience", "2",
                "--bpr-factors", "3",
                "--bpr-epochs", "1",
                "--bpr-batch-size", "2",
                "--factors", "2",
                "--epochs", "2",
            ]
            args = system._parser().parse_args(argv)
            system._validate_args(args, system._parser())
            with patch.object(system, "load_spacy_model", return_value=FakeNLP()), patch.object(
                system, "build_aspect_profiles", return_value=fake_profiles
            ):
                result = system.run(args)

            self.assertIn(result["summary"]["selected_model"], result["metrics"])
            for filename in (
                "dev_metrics.json",
                "ablation.json",
                "aspect_diagnostics.json",
                "error_analysis.json",
                "test_recommendations.csv",
                "test_recommendations_mandatory.csv",
                "trace.jsonl",
                "trace_best.jsonl",
                "mf_tuning.json",
                "bpr_training.json",
            ):
                self.assertTrue((out / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
