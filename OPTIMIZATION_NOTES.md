# Performance and resume changes

This revision preserves the assignment's mandatory full-data, CPU, spaCy/VADER,
five-aspect, sparse-matrix, deterministic, and train-only fitting rules.

## Changes

- `--resume` validates fingerprints and reuses completed Phase 0, aspect-profile,
  sparse-rating, and matrix-factorization checkpoints from `results/cache`.
- spaCy still uses `nlp.pipe(..., batch_size=1000, n_process=K)` as required.
- `aspect_evidence.jsonl` is capped by `--evidence-limit` because it is diagnostic,
  while every training review still contributes to profiles and summary counts.
- CF and aspect scores are vectorized over the catalogue with NumPy.
- Baselines and the alpha sweep reuse one component pass per user instead of
  repeatedly scoring every user-item pair.
- Phase 0 and NLP progress messages are flushed to the run log.

## Validation

- All mandatory and advanced synthetic tests pass, including an end-to-end
  orchestration test for sweep, early stopping, BPR, automatic selection, and
  output generation.
- Cache round trips for sparse ratings, aspect profiles, and MF arrays are tested.
- Multi-configuration rankings are tested against the single-configuration path.
- A synthetic ranking-kernel benchmark (30 users, 1,500 items, six settings)
  measured 10.835 seconds for the former scalar pattern and 0.113 seconds for the
  vectorized/reused implementation on the validation environment. This 96.3x
  kernel speedup is not an estimate of total full-run speed, which remains dominated
  by the required spaCy parse and raw-file scan.

Do not submit sampled-data metrics. Sampling may be used only for a smoke test;
the final command must process the complete retained Philadelphia training split.
