import json
import sys
from pathlib import Path

import nltk
import numpy
import pandas
import pytest
import scipy
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

EXPECTED_FILES = [
    "system.py",
    "data_preprocessor.py",
    "text_processor.py",
    "sentiment.py",
    "hybrid_model.py",
    "evaluator.py",
    "advanced_nlp.py",
    "advanced_models.py",
    "config.json",
    "requirements.txt",
    "requirements-advanced.txt",
    "requirements-transformer.txt",
]

EXPECTED_DIRECTORIES = [
    "tests",
    "results",
    "extra",
    "data/raw",
    "data/processed",
]


@pytest.fixture(scope="session")
def config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def nlp():
    return spacy.load("en_core_web_md")


def test_python_version():
    assert sys.version_info[:2] == (3, 11)
    assert sys.version_info.releaselevel == "final"


def test_dependency_versions():
    assert spacy.__version__ == "3.7.5"
    assert nltk.__version__ == "3.9.2"
    assert numpy.__version__ == "1.26.4"
    assert pandas.__version__ == "2.2.2"
    assert scipy.__version__ == "1.13.1"
    assert pytest.__version__ == "8.2.2"


def test_spacy_model(nlp):
    assert nlp.meta["version"] == "3.7.1"
    assert nlp.vocab.vectors_length == 300
    assert {"tagger", "parser", "lemmatizer"}.issubset(nlp.pipe_names)


def test_vader_resource():
    nltk.data.find("sentiment/vader_lexicon.zip")

    analyzer = SentimentIntensityAnalyzer()
    positive = analyzer.polarity_scores("The food was excellent.")
    negative = analyzer.polarity_scores("The service was terrible.")

    assert positive["compound"] > 0.5
    assert negative["compound"] < -0.4


def test_project_structure():
    missing_files = [
        path for path in EXPECTED_FILES
        if not (ROOT / path).is_file()
    ]
    missing_directories = [
        path for path in EXPECTED_DIRECTORIES
        if not (ROOT / path).is_dir()
    ]

    assert not missing_files, f"Missing files: {missing_files}"
    assert not missing_directories, (
        f"Missing directories: {missing_directories}"
    )


def test_config_values(config):
    assert config["project"]["seed"] == 42

    assert config["data"]["city"] == "Philadelphia"
    assert config["data"]["split_date"] == "2021-01-01"
    assert config["data"]["min_interactions"] == 5
    assert config["data"]["min_train_rows_for_dev_user"] == 3
    assert config["data"]["category_labels"] == [
        "Restaurants",
        "Food",
    ]

    assert config["nlp"]["model"] == "en_core_web_md"
    assert config["nlp"]["model_version"] == "3.7.1"
    assert config["nlp"]["aspects"] == [
        "food",
        "service",
        "ambiance",
        "price",
        "location",
    ]
    assert config["nlp"]["similarity_threshold"] == 0.35
    assert config["nlp"]["batch_size"] == 1000
    assert config["nlp"]["n_process"] == 4

    assert config["sentiment"]["method"] == "vader"
    assert config["sentiment"]["epsilon"] == 1e-9

    assert config["matrix_factorization"]["fit_split"] == "train"
    assert config["hybrid"]["alpha"] == 0.7
    assert config["hybrid"]["normalization"] == "global"

    assert config["ranking"]["top_k"] == 10
    assert config["ranking"]["fallback_m"] == 5
    assert config["ranking"]["abstention_threshold"] == 0.3
    assert config["ranking"]["exclude_seen_items"] is True

    assert config["evaluation"]["k"] == 10
    assert config["evaluation"]["trace_test_users"] == 20
