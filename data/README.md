# Sample Data

`sample_feedback.csv` contains six fictional reviews and twelve aspect-level annotations in long form. It is used for dataset import, statistics, and evaluation examples.

The original `reviews_multi_aspect_merged_long.csv` and `feedback.db` are intentionally excluded. To evaluate a locally available dataset with the original schema, pass its path explicitly:

```bash
python evaluate_sentiment.py /path/to/reviews_multi_aspect_merged_long.csv
```

Do not treat the included sample's accuracy as a benchmark.
