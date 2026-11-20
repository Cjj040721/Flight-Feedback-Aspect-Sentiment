# -----------------------------------------------------------------------------
# run_server.py
# Laboratory: Lab1, NLP Coursework
# Date: 2025-11-19
#
# This module initializes and launches the web server for the feedback analysis
# system. It configures runtime environment variables for stability on macOS,
# CPU/MPS environments, and multiprocessing contexts. It also performs model
# warmup to avoid concurrency issues caused by on-demand model loading.
#
# The server hosts REST endpoints defined in `web.app`, enabling aspect
# detection and sentiment analysis to be accessed interactively or by external
# clients during experiments.
# -----------------------------------------------------------------------------

import os, multiprocessing as mp, pathlib

# -----------------------------------------------------------------------------
# Environment configuration for Transformer and Torch runtimes
# -----------------------------------------------------------------------------
# Enable MPS fallback on macOS systems where GPU kernels may not support all ops.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Disable CUDA explicitly—this system is CPU-first and must remain stable
# across machines without NVIDIA hardware.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disable tokenizer parallelism to avoid multi-thread contention and warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Limit OpenMP and MKL threading to 1 for reproducibility and to avoid
# oversubscription when using multiprocessing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure HuggingFace models use a consistent cache directory.
os.environ.setdefault("HF_HOME", str(pathlib.Path.home() / ".cache" / "huggingface"))

# Redundant but harmless safeguard for macOS MPS stability.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# -----------------------------------------------------------------------------
# Multiprocessing configuration
# -----------------------------------------------------------------------------
# Use the "spawn" start method, which avoids CUDA/MPS and fork-related issues
# when loading Transformers or Torch models across processes. This is the
# recommended method for macOS and for any code involving model loading.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method already set; safe to ignore
    pass

# -----------------------------------------------------------------------------
# Application import
# -----------------------------------------------------------------------------
from web import app
from aspects import detect_aspects
from sentiment import overall_sentiment_detail

# -----------------------------------------------------------------------------
# Pre-warming models
# -----------------------------------------------------------------------------
# Warmup ensures that:
#   (1) models are loaded in the main process
#   (2) concurrent loading is avoided
#   (3) first request latency is reduced
#
# Both aspect detection (zero-shot classifier) and sentiment engines are warmed.
print("🔥 warming up models ...")
detect_aspects("加载很慢，客服不在线，支付页有附加费。", use_zero_shot=True)
overall_sentiment_detail("整体体验还不错")
print("✅ warmup done")

# -----------------------------------------------------------------------------
# Launch server
# -----------------------------------------------------------------------------
# For evaluation and development, we use:
#   - host: localhost
#   - single-process, single-threaded mode
#   - debug=False to avoid reloader conflicts with multiprocessing
#
# The configuration ensures reproducibility and eliminates race conditions
# when dealing with heavy NLP model pipelines.
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False
    )
