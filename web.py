# -----------------------------------------------------------------------------
# web.py
# Laboratory: Lab1, NLP Coursework
# Date: 2025-11-19
#
# Flask-based Web UI for the multi-aspect sentiment analyzer.
#
# This module exposes:
#   - A browser UI ("/") for interactive feedback submission, aspect detection,
#     and aspect-level sentiment visualization.
#   - A JSON API endpoint ("/api/summary") for aggregated aspect statistics.
#
# The design emphasizes stability on macOS/ARM environments and reproducible
# experiments, especially when using HuggingFace Transformers and SQLite.
# -----------------------------------------------------------------------------

# Flask Web UI for the analyzer (stable on macOS/ARM)

import os, multiprocessing as mp, pathlib

# ---- Environment stability settings (must be configured before imports) ----
# 环境稳定性设置（必须放最前面）

# Disable tokenizer parallelism to avoid thread contention and occasional
# crashes with certain tokenizers on macOS.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Limit the number of threads used by OpenMP and MKL to 1, which:
#   - reduces interference with multiprocessing
#   - improves reproducibility of latency and CPU usage
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# If models are fully cached locally, offline mode can be enabled by:
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 若本地已缓存模型，可开启离线模式（缓存目录通常 ~/.cache/huggingface/hub）

# Set HuggingFace cache directory to a deterministic location in the user home.
os.environ.setdefault(
    "HF_HOME",
    str(pathlib.Path.home() / ".cache" / "huggingface")
)

from flask import Flask, request, render_template_string, jsonify
from transformers.utils import logging as hf_logging

# Reduce verbosity of HuggingFace logs to warnings to keep the web UI clean.
hf_logging.set_verbosity_warning()

from aspects import detect_aspects, ASPECT_KEYWORDS, ASPECT_DISPLAY
from sentiment import aspect_sentiment
from storage import (
    init_db,
    insert_feedback,
    insert_aspect_opinions,
    fetch_recent,
    fetch_all_aspects,
)
from analysis import compute_aspect_stats

# -----------------------------------------------------------------------------
# HTML template (minimal, self-contained)
# -----------------------------------------------------------------------------
# The template uses sakura.css (a lightweight CSS framework) to keep the UI
# simple yet readable. It provides:
#   - a text area to paste feedback
#   - a table summarizing last analysis results
#   - a summary table over all feedback
#   - a list of recent feedback entries
TEMPLATE = """
<!doctype html>
<title>Flight feedback analyzer</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/sakura.css/css/sakura.css">
<div class="container">
  <h2>Customer feedback analyzer</h2>
  <form method="post">
    <textarea name="text" rows="4" style="width:100%;" placeholder="Paste a feedback...">{{ text or "" }}</textarea>
    <button type="submit">Analyze & Save</button>
  </form>

  {% if last_result %}
  <h3>Last result</h3>
  <div class="card">
    <p>{{ last_result.text }}</p>
    <table>
      <thead><tr><th>Aspect</th><th>Sentiment</th><th>Score</th></tr></thead>
      <tbody>
        {% for op in last_result.opinions %}
          <tr>
            <td>{{ op.aspect }}</td>
            <td>{{ op.label }}</td>
            <td>{{ "%.3f"|format(op.score) }}</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  <h3>Summary (all feedback)</h3>
  <table>
    <thead>
      <tr><th>Aspect</th><th>Total</th><th>Positive</th><th>Negative</th><th>Neutral</th></tr>
    </thead>
    <tbody>
      {% for row in summary_rows %}
        <tr>
          <td>{{ row.aspect }}</td>
          <td>{{ row.total }}</td>
          <td>{{ row.ratios.positive }}%</td>
          <td>{{ row.ratios.negative }}%</td>
          <td>{{ row.ratios.neutral }}%</td>
        </tr>
      {% endfor %}
    </tbody>
  </table>

  <h3>Recent feedback</h3>
  {% for r in recent %}
    <div class="card">
      <div><small>#{{ r.id }} | {{ r.created_at }}</small></div>
      <div>{{ r.text }}</div>
    </div>
  {% endfor %}
</div>
"""

# -----------------------------------------------------------------------------
# Flask app factory
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# Main UI route
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    """
    Web UI entry point.

    Behavior
    --------
    - GET:
        * Initialize DB schema (idempotent)
        * Render summary statistics and recent feedback
    - POST:
        * Read submitted feedback text
        * Run aspect detection and aspect-level sentiment
        * Persist feedback + aspect-opinion records
        * Recompute and display summary

    Notes
    -----
    - init_db() is called at each request for robustness; it is cheap due
      to IF NOT EXISTS in the schema.
    """
    init_db()
    last_result = None
    text = ""

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        if text:
            # Insert raw feedback
            fid = insert_feedback(text)

            # Aspect detection + aspect-level sentiment analysis
            ops = []
            aspects = detect_aspects(text)
            for a in aspects:
                lab, score, eng = aspect_sentiment(text, ASPECT_KEYWORDS.get(a, []))
                ops.append(
                    {
                        "aspect": a,
                        "label": lab,
                        "score": float(score),
                        "engine": eng,
                    }
                )

            # Persist aspect-level opinions using the normalized sentiment label
            insert_aspect_opinions(
                fid,
                [
                    {
                        "aspect": o["aspect"],
                        "sentiment": o["label"],
                        "score": o["score"],
                        "source": "user_auto",
                    }
                    for o in ops
                ],
            )

            last_result = {"text": text, "opinions": ops}

    # Compute global summary statistics over all stored aspect opinions
    rows = fetch_all_aspects()
    stats = compute_aspect_stats(rows)
    ordered = sorted(stats.items(), key=lambda kv: kv[1]["total"], reverse=True)

    summary_rows = [
        {
            "aspect": ASPECT_DISPLAY.get(k, k),
            "total": v["total"],
            "ratios": v["ratios"],
        }
        for k, v in ordered
    ]

    # Fetch recent feedback entries for display
    recent = fetch_recent(limit=10)

    return render_template_string(
        TEMPLATE,
        text=text,
        recent=recent,
        last_result=last_result,
        summary_rows=summary_rows,
    )

# -----------------------------------------------------------------------------
# JSON API: global aspect summary
# -----------------------------------------------------------------------------
@app.route("/api/summary")
def api_summary():
    """
    Return aggregated aspect-level sentiment statistics as JSON.

    Returns
    -------
    JSON
        Structure equivalent to the output of analysis.compute_aspect_stats().
    """
    rows = fetch_all_aspects()
    return jsonify(compute_aspect_stats(rows))

# -----------------------------------------------------------------------------
# Standalone server entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # macOS/ARM stability: use 'spawn' start method and avoid debug/reloader/threads.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set; safe to ignore.
        pass

    # Single warmup call to pre-load models and avoid mmap-related race
    # conditions on the first concurrent request.
    try:
        detect_aspects("加载很慢，客服不在线，支付页有附加费。", use_zero_shot=True)
    except Exception as e:
        print("Warmup warning:", e)

    # Run the Flask development server in a controlled, single-threaded mode.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False,
    )
