# -----------------------------------------------------------------------------
# evaluate_sentiment.py
# Laboratory: Lab1, NLP Coursework
# Date: 2025-11-19
#
# This script evaluates the accuracy of the system's overall sentiment
# classifier using a CSV file containing gold-standard annotations. The
# evaluation protocol aggregates multiple human labels per review (if present)
# using majority voting, then compares the system’s predictions against the
# aggregated gold labels.
# -----------------------------------------------------------------------------

import csv
from typing import List, Tuple
from sentiment import overall_sentiment

# Allowed sentiment labels used for validation and majority voting.
LABELS = ["positive", "negative", "neutral"]

# -----------------------------------------------------------------------------
# load_gold_from_csv
# -----------------------------------------------------------------------------
def load_gold_from_csv(path: str) -> Tuple[List[str], List[str]]:
    """
    Load gold-standard sentiment annotations from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file containing review_id, text, sentiment, and source.
        Only rows where `source == "gold"` are considered gold annotations.

    Returns
    -------
    Tuple[List[str], List[str]]
        A pair (texts, labels):
            - texts: list of unique review texts
            - labels: list of corresponding gold labels obtained via majority vote

    Details
    -------
    - Multiple gold labels may exist for the same review_id (e.g., multiple
      annotators). We group all labels under the same review_id and apply
      majority voting (most frequent label wins).
    - Only entries whose sentiment label is in LABELS are used.
    - Reviews without valid text or label are ignored.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only accept entries explicitly marked as gold-standard
            if (row.get("source") or "").strip().lower() != "gold":
                continue

            rid = row.get("review_id") or ""
            txt = (row.get("text") or "").strip()
            lab = (row.get("sentiment") or "").strip().lower()

            # Keep only valid labeled instances
            if not txt or lab not in LABELS:
                continue

            rows.append((rid, txt, lab))

    # Group labels by review ID
    byid = {}
    for rid, txt, lab in rows:
        if rid not in byid:
            byid[rid] = {"text": txt, "labels": []}
        byid[rid]["labels"].append(lab)

    # Construct text list and majority-vote gold label list
    texts, labels = [], []
    for rid, v in byid.items():
        texts.append(v["text"])
        labs = v["labels"]

        # Majority vote over ["positive", "negative", "neutral"]
        counts = {k: labs.count(k) for k in LABELS}
        # Select label with highest count
        labels.append(max(counts.items(), key=lambda kv: kv[1])[0])

    return texts, labels

# -----------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------
def evaluate(path: str):
    """
    Compute sentiment classification accuracy on a gold-standard dataset.

    Parameters
    ----------
    path : str
        Path to the dataset CSV file.

    Process
    -------
    1. Load gold-standard text and majority-vote labels.
    2. Predict system labels using `overall_sentiment`.
    3. Compute simple accuracy = correct_predictions / total_samples.

    Notes
    -----
    - Accuracy is a coarse metric; in real evaluations, one may additionally
      compute macro-F1, confusion matrices, or per-class precision/recall.
    """
    texts, gold = load_gold_from_csv(path)
    preds = [overall_sentiment(t) for t in texts]

    n = len(gold)
    correct = sum(1 for g, p in zip(gold, preds) if g == p)
    acc = correct / n if n else 0.0

    print(f"Accuracy = {acc:.4f} ({correct}/{n})")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    evaluate(sys.argv[1] if len(sys.argv) > 1 else "data/sample_feedback.csv")
