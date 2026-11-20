# Validation Record

Last updated: 2026-11-20.

## Source Cleanup

- Extracted only application source from the recovered ZIP.
- Excluded `.venv`, `.idea`, `.pytest_cache`, `__pycache__`, AppleDouble files, and `data/feedback.db`.
- Kept the original review CSV outside the public repository because its redistribution terms were not confirmed.
- Added six fictional bilingual reviews for import and evaluation workflows.

## Fixes Verified During Cleanup

1. Removed the final keyword-only filter that discarded valid zero-shot predictions and forced `OTHER`.
2. Replaced English substring matching with boundary matching for single words.
3. Added near-top-score filtering after observing that multi-label NLI assigned high scores to unrelated aspects in the same sentence.
4. Added English `but` and Chinese contrast segmentation for aspect-specific sentiment.
5. Made `HF_HOME` overridable and retained CPU/macOS startup safeguards.
6. Made SQLite work with a filename-only `FEEDBACK_DB_PATH` and replaced deprecated UTC timestamp creation.
7. Updated `sentencepiece` to `0.2.0` to use an Apple Silicon prebuilt wheel instead of requiring a local CMake build.

## Automated Checks

`python -m unittest discover -s tests -v` — **8 passed**.

The suite covers zero-shot predictions without keywords, near-top-score selection, model-failure fallback, English token boundaries, contrast splitting, SQLite statistics, Web submission, JSON summary, and fictional CSV import/evaluation loading.

## Real Model and CLI Evidence

The following models downloaded and ran successfully:

- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- `uer/roberta-base-finetuned-jd-binary-chinese`
- `distilbert-base-uncased-finetuned-sst-2-english`

Verified English CLI output:

```text
SITE_SPEED        positive  score=0.798
PAYMENT           negative  score=0.200
```

Verified Chinese CLI output:

```text
SITE_SPEED        negative  score=0.118
CUSTOMER_SERVICE  positive  score=0.604
```

Recent-history and aggregate-summary commands also returned the persisted records.

## Real Browser Evidence

A real `run_server.py` process was used with Chromium/Playwright. The browser submitted the same English and Chinese examples and rendered the same labels and scores as the CLI. Aggregate counts updated correctly after both submissions.

- `docs/assets/web-ui-english.png`
- `docs/assets/web-ui-bilingual.png`

The server was stopped after verification. Runtime verification databases were removed.

## Evaluation Limit

The evaluation command completed on the included fictional sample and returned `4/6`. This only verifies the evaluation path; it is not a statistically meaningful model-quality score. The approximate accuracy in the original report was explicitly marked as a placeholder and is not used here.
