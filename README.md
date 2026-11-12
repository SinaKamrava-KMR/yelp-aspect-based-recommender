# NLP HW3 — Optimized Aspect-Based Hybrid Recommender

This revision preserves the exact mandatory Assignment 3 system and adds every
optimization as an independently named experiment.  The mandatory result is never
silently replaced by a bonus method.

## Implemented methods

- Exact mandatory spaCy mapping and dependency-guided, unmodified NLTK VADER
- Safe chunking of pathological long reviews without sampling or dropping text
- Full dev sweep of `alpha`, fallback `M`, and abstention threshold `tau`
- Aspect coverage, ambiguity, mapped-term, evidence, and star-correlation diagnostics
- Advanced dependency sentiment: verbs, nominal predicates, conjunctions,
  intensifiers, local negation, and contrast
- Confidence-weighted evidence and empirical-Bayes shrinkage
- Train-only Word2Vec aspect mapping
- Matrix-factorization grid search with dev-only early stopping
- Popularity-aware and evidence-adaptive hybrid rankers
- BPR-MF optimized directly for ranking
- Optional Transformer ABSA ablation
- Interaction-bucket error analysis and deterministic model selection
- Independent checkpoints for every expensive stage

## Files

- `system.py`: one-command mandatory and advanced orchestration
- `data_preprocessor.py`: deterministic streaming Phase 0
- `text_processor.py`: aspect mapping, profiles, diagnostics, and safe chunking
- `sentiment.py`: mandatory and advanced dependency/VADER scorers
- `hybrid_model.py`: sparse ratings, biased MF, normalization, and rankers
- `advanced_models.py`: MF tuning and BPR-MF
- `advanced_nlp.py`: train-only Word2Vec and optional Transformer ABSA
- `evaluator.py`: RMSE, HR@10, and NDCG@10
- `tests/`: mandatory, advanced, checkpoint, and integration tests

## Windows PowerShell setup

Run these commands from the project root with the confirmed Python 3.11.9
environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-advanced.txt
python -m pip install .\en_core_web_md-3.7.1-py3-none-any.whl
python -m nltk.downloader vader_lexicon
python -m pytest -v
```

If the existing `.venv` is already correct, activate it and run only the install
and test commands.  `requirements-advanced.txt` includes the mandatory pins.

## Recommended complete run (CPU-safe)

This runs the mandatory system plus the full parameter sweep, enhanced VADER,
confidence weighting, shrinkage, MF tuning, popularity-aware/adaptive rankers,
BPR, and Word2Vec.  It reuses the successful mandatory checkpoints already under
`results/cache`.

```powershell
python -u .\system.py `
  --raw-dir ".\data\raw" `
  --train-reviews ".\data\processed\train_reviews.csv" `
  --businesses ".\data\processed\businesses.jsonl" `
  --dev-reviews ".\data\processed\dev_reviews.csv" `
  --dev-qrels ".\data\processed\dev_qrels.tsv" `
  --dev-users ".\data\processed\dev_users.csv" `
  --test-users ".\data\raw\test_users.csv" `
  --outdir ".\results" `
  --alpha 0.7 `
  --n-process 1 `
  --batch-size 1000 `
  --min-interactions 5 `
  --threshold 0.3 `
  --seed 42 `
  --resume `
  --run-all-experiments `
  --evidence-limit 200 `
  --progress-every 10000 `
  2>&1 | Tee-Object -FilePath ".\optimized_run.log"
```

`--n-process 1` is recommended because the previous four-process run exhausted
RAM once.  The assignment-required `batch_size=1000` remains unchanged and all
retained Philadelphia reviews are processed.

## Optional Transformer stage

Run this only after the CPU-safe command completes.  All earlier checkpoints are
reused; only the Transformer profile and its evaluation are added.

```powershell
python -m pip install -r requirements-transformer.txt
```

```powershell
python -u .\system.py `
  --raw-dir ".\data\raw" `
  --train-reviews ".\data\processed\train_reviews.csv" `
  --businesses ".\data\processed\businesses.jsonl" `
  --dev-reviews ".\data\processed\dev_reviews.csv" `
  --dev-qrels ".\data\processed\dev_qrels.tsv" `
  --dev-users ".\data\processed\dev_users.csv" `
  --test-users ".\data\raw\test_users.csv" `
  --outdir ".\results" `
  --alpha 0.7 `
  --n-process 1 `
  --batch-size 1000 `
  --min-interactions 5 `
  --threshold 0.3 `
  --seed 42 `
  --resume `
  --run-all-experiments `
  --use-transformer `
  --transformer-device auto `
  --evidence-limit 200 `
  --progress-every 10000 `
  2>&1 | Tee-Object -FilePath ".\transformer_run.log"
```

The default ABSA checkpoint is
`yangheng/deberta-v3-base-absa-v1.1`.  A CUDA GPU is strongly preferred.  On a
CPU this stage can be much slower than the rest of the project.

## Outputs

Mandatory-compatible outputs:

- `results/dev_recommendations.csv`
- `results/test_recommendations.csv`
- `results/test_recommendations_mandatory.csv`
- `results/dev_metrics.json`
- `results/summary.json`
- `results/trace.jsonl`

Advanced evidence:

- `results/ablation.json`: complete sweep and model comparison
- `results/aspect_diagnostics.json`: coverage and aspect error diagnostics
- `results/error_analysis.json`: cold/warm/heavy user results
- `results/mf_tuning.json`: all MF trials and early-stopping history
- `results/bpr_training.json`: BPR settings and loss history
- `results/trace_best.jsonl`: trace for the automatically selected model
- `results/aspect_evidence_*.jsonl`: capped human-readable evidence samples

Model selection uses dev `NDCG@10` first, then `HR@10`, then `RMSE`, with a
deterministic name tie-break.  Test users never influence training or selection.
The mandatory `trace.jsonl` always describes the assignment-exact VADER hybrid;
`trace_best.jsonl` describes the selected advanced model.

## Resume behavior

Do not delete `results/cache`.  The mandatory cache key remains compatible with
the already successful full run.  Every advanced variant has its own fingerprint,
so changing Word2Vec, Transformer, shrinkage, or MF parameters invalidates only
the affected stage.
