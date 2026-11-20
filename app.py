# -----------------------------------------------------------------------------
# app.py
# Laboratory: Lab1, NLP Coursework
# Date: 2025-11-19
#
# This script provides a command-line interface (CLI) for a multi-model
# flight-booking feedback analysis system. It supports feedback ingestion,
# aspect extraction, aspect-level sentiment analysis, database storage,
# dataset import, and summary reporting. The CLI interacts with underlying
# NLP components and a persistence layer to enable end-to-end experimentation.
# -----------------------------------------------------------------------------

import argparse
from typing import List, Dict
from aspects import ASPECT_KEYWORDS, detect_aspects
from sentiment import aspect_sentiment
from storage import (
    init_db, reset_db, insert_feedback, insert_aspect_opinions,
    fetch_recent, fetch_all_aspects
)
from analysis import compute_aspect_stats, pretty_table
from importer import import_dataset


def add_feedback_cli(text: str):
    """
    Add a single piece of textual feedback into the system, perform
    automatic aspect detection and aspect-level sentiment analysis,
    then store results in the database.

    Parameters
    ----------
    text : str
        Raw user feedback to be processed.

    Notes
    -----
    - Each detected aspect is processed independently using keyword-guided
      sentiment classification.
    - Sentiment model outputs include the predicted label, a confidence score,
      and the model engine name; only label and score are stored here.
    """
    fid = insert_feedback(text)
    opinions: List[Dict] = []

    # Detect relevant aspects from the input text
    aspects = detect_aspects(text)
    for a in aspects:
        keywords = ASPECT_KEYWORDS.get(a, [])
        label, score, engine = aspect_sentiment(text, keywords)

        # Store the opinion structure for later database insertion
        opinions.append({
            "aspect": a,
            "sentiment": label,
            "score": float(score),
            "source": "user_auto"
        })

    insert_aspect_opinions(fid, opinions)

    print(f"[OK] Added feedback #{fid}")
    for op in opinions:
        print(f"  - {op['aspect']:16s}  {op['sentiment']:8s}  score={op['score']:.3f}")


def do_list(limit: int = 10):
    """
    Display the most recent feedback entries.

    Parameters
    ----------
    limit : int
        Maximum number of feedback items to display (default: 10).

    Notes
    -----
    - Feedback entries include ID, timestamp, optional external Review ID,
      and full feedback text.
    """
    items = fetch_recent(limit=limit)
    if not items:
        print("(empty)")
        return

    for it in items:
        print(f"#{it['id']:04d}  {it['created_at']}  {it.get('review_id') or ''}")
        print("   ", it['text'])
        print()


def do_summary():
    """
    Compute and print aspect-level sentiment distributions using aggregated
    database entries.

    Notes
    -----
    - Fetches all aspect-sentiment annotations stored in the system.
    - Uses analysis.compute_aspect_stats() to compute ratios and totals.
    - pretty_table() renders results in a formatted ASCII table.
    """
    rows = fetch_all_aspects()
    stats = compute_aspect_stats(rows)
    print(pretty_table(stats))


def do_load(path: str):
    """
    Load and ingest an external dataset (JSONL or long-form CSV).

    Parameters
    ----------
    path : str
        Filesystem path to the dataset.

    Notes
    -----
    - The importer handles normalization and database insertion.
    - Intended for bulk loading during experiments.
    """
    import_dataset(path)


def do_reset():
    """
    Reset the entire database, erasing all stored feedback and analyses.

    Warning
    -------
    This action is destructive and cannot be undone.
    """
    reset_db()
    print("[OK] database reset.")


def main():
    """
    Configure and execute the command-line interface.

    Notes
    -----
    - Provides subcommands:
        * add            — Add one feedback text and auto-analyze it
        * list           — View recent feedback entries
        * summary        — Print aspect-level sentiment statistics
        * load-dataset   — Import a dataset for batch processing
        * reset          — Reset the database
    - Automatically initializes the database on startup.
    """
    init_db()

    parser = argparse.ArgumentParser(
        description="Flight-booking feedback analyzer (multi-model)"
    )
    sub = parser.add_subparsers()

    # Add feedback
    a = sub.add_parser("add", help="Add one feedback text")
    a.add_argument("text", nargs="+", help="feedback text")
    a.set_defaults(func=lambda args: add_feedback_cli(" ".join(args.text)))

    # List feedback
    l = sub.add_parser("list", help="List recent feedback")
    l.add_argument("--limit", type=int, default=10)
    l.set_defaults(func=lambda args: do_list(args.limit))

    # Summary of aspect-level sentiment
    s = sub.add_parser("summary", help="Print aspect ratios")
    s.set_defaults(func=lambda args: do_summary())

    # Load external dataset
    ld = sub.add_parser("load-dataset",
                        help="Load dataset (jsonl/csv-long)")
    ld.add_argument("path",
                    help="path to reviews_multi_aspect_merged.jsonl or _long.csv")
    ld.set_defaults(func=lambda args: do_load(args.path))

    # Reset database
    r = sub.add_parser("reset", help="Reset database")
    r.set_defaults(func=lambda args: do_reset())

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
