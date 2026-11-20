# -----------------------------------------------------------------------------
# sentiment.py
# Laboratory: Lab1, NLP Coursework
# Date: 2025-11-19
#
# Multi-model sentiment classification module.
#
# This module implements a hybrid sentiment analyzer combining:
#   (1) English Transformer classifier (DistilBERT SST-2)
#   (2) Chinese Transformer classifier (RoBERTa-JD Chinese binary sentiment)
#   (3) SnowNLP (for Chinese only; unreliable for English)
#
# The system unifies these sources through a weighted ensemble, supported by:
#   - A neutral probability band to reduce overconfident misclassifications
#   - Clause-level segmentation and aspect-focused sentiment extraction
#   - Thread-safe lazy loading of Transformers pipelines
#
# The design aims to handle bilingual (EN + Chinese) feedback reliably, which
# is typical in mixed-language user reviews for travel/booking platforms.
# -----------------------------------------------------------------------------

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# LOCAL_ZH_MODEL = "models/zh"
# LOCAL_EN_MODEL = "models/en"

from typing import Tuple, List
import re
from threading import Lock

# Canonical label set used throughout the system
LABELS = ("positive", "negative", "neutral")

# -----------------------------------------------------------------------------
# Lazy initialization of sentiment pipelines (thread-safe)
# -----------------------------------------------------------------------------
# _ZH_CLF : Chinese binary classifier
# _EN_CLF : English binary classifier
#
# These are loaded lazily since loading Transformer pipelines is expensive and
# should not occur in worker subprocesses or during concurrent requests.
_ZH_CLF = None
_EN_CLF = None
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CLF_LOCK = Lock()

# -----------------------------------------------------------------------------
# Keyword matching for clause-level filtering (aspect-specific)
# -----------------------------------------------------------------------------
def _clause_has_keyword(clause: str, kws: List[str]) -> bool:
    """
    Robust clause-level keyword detection used by aspect-window sentiment.

    Rules
    -----
    - Chinese keywords:
        Direct substring matching (no whitespace segmentation in CJK)
    - English keywords:
        Strict word-boundary match (\\bword\\b) to avoid spurious hits
        such as:
            'breakfast' → falsely matching 'fast'
            'repayment' → falsely matching 'pay'

    Parameters
    ----------
    clause : str
        The text clause being checked.
    kws : List[str]
        Aspect-specific list of candidate keywords.

    Returns
    -------
    bool
        True if clause contains at least one keyword.
    """
    if not clause or not kws:
        return False
    lc = clause.lower()

    for k in kws:
        if not k:
            continue
        k = k.lower()

        # Chinese keyword: substring match
        if _CJK_RE.search(k):
            if k in clause:
                return True

        # English keyword: strict full-word match
        else:
            pattern = r"\b{}\b".format(re.escape(k))
            if re.search(pattern, lc):
                return True

    return False

# -----------------------------------------------------------------------------
# Lazy loading of Transformers sentiment pipelines
# -----------------------------------------------------------------------------
def _get_pipelines():
    """
    Load and return the Chinese and English sentiment-analysis pipelines.

    Thread Safety
    -------------
    Uses a global lock to ensure only one thread loads models.
    After a failure, stores False to avoid repeated load attempts.

    Returns
    -------
    (zh_clf, en_clf)
        Either pipeline objects or None (if failed).
    """
    global _ZH_CLF, _EN_CLF
    if _ZH_CLF is not None and _EN_CLF is not None:
        return _ZH_CLF if _ZH_CLF else None, _EN_CLF if _EN_CLF else None

    with _CLF_LOCK:
        if _ZH_CLF is not None and _EN_CLF is not None:
            return _ZH_CLF if _ZH_CLF else None, _EN_CLF if _EN_CLF else None

        try:
            from transformers import pipeline
        except Exception:
            _ZH_CLF = _EN_CLF = False
            return None, None

        # Chinese model (JD review dataset)
        try:
            _ZH_CLF = pipeline(
                "sentiment-analysis",
                model="uer/roberta-base-finetuned-jd-binary-chinese",
                device=-1      # enforce CPU for reproducibility
            )
        except Exception:
            _ZH_CLF = False

        # English model (SST-2)
        try:
            _EN_CLF = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
        except Exception:
            _EN_CLF = False

        return _ZH_CLF if _ZH_CLF else None, _EN_CLF if _EN_CLF else None

# -----------------------------------------------------------------------------
# Language detection: simple CJK presence check
# -----------------------------------------------------------------------------
def _is_chinese(text: str) -> bool:
    """
    Detect whether input contains Chinese characters.

    Notes
    -----
    This is a lightweight heuristic, but effective for routing text to the
    appropriate sentiment model (SnowNLP or Chinese Transformers).
    """
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

# -----------------------------------------------------------------------------
# SnowNLP sentiment probability
# -----------------------------------------------------------------------------
def _snownlp_prob(text: str) -> float:
    """
    Compute sentiment using SnowNLP.

    Notes
    -----
    - SnowNLP is specialized for Chinese; it is unreliable for English:
          → In English texts, we return a neutral default: 0.5
    - Failure to load SnowNLP also results in returning 0.5.

    Returns
    -------
    float in [0,1]
        Probability of positive sentiment.
    """
    # Only apply to Chinese text
    if not _is_chinese(text):
        return 0.5

    try:
        from snownlp import SnowNLP
        return float(SnowNLP(text).sentiments)
    except Exception:
        return 0.5

# -----------------------------------------------------------------------------
# Neutral band and model weights
# -----------------------------------------------------------------------------
# The neutral region is intentionally narrow to reduce ambiguous cases.
NEU_LOW, NEU_HIGH = 0.48, 0.52

# Ensemble weighting:
# - English text → SnowNLP gives 0.5 neutral baseline, thus Transformer dominates
# - Chinese text → both models contribute
W_XF, W_SNOW = 0.6, 0.4

# -----------------------------------------------------------------------------
# Transformer sentiment probability
# -----------------------------------------------------------------------------
def _xf_prob(text: str) -> float:
    """
    Compute sentiment probability using Transformer models.

    Routing Logic
    -------------
    - If text contains Chinese and the Chinese classifier is available → use it.
    - Else if English classifier is available → use it.
    - Else return neutral probability 0.5.

    Returns
    -------
    float in [0,1]
        Estimated probability of positive sentiment.
    """
    zh, en = _get_pipelines()

    # Chinese branch
    if _is_chinese(text) and zh:
        try:
            pred = zh(text)[0]
            p = float(pred["score"])
            return p if "pos" in pred["label"].lower() else 1.0 - p
        except Exception:
            pass

    # English branch
    if en:
        try:
            pred = en(text)[0]
            p = float(pred["score"])
            return p if "pos" in pred["label"].lower() else 1.0 - p
        except Exception:
            pass

    # Fallback
    return 0.5

# -----------------------------------------------------------------------------
# Prob → Label mapping with neutral band
# -----------------------------------------------------------------------------
def _label_from_prob(p: float) -> str:
    """
    Convert probability into {positive, negative, neutral},
    applying a neutral band to dampen borderline predictions.
    """
    if p < NEU_LOW:
        return "negative"
    if p > NEU_HIGH:
        return "positive"
    return "neutral"

# -----------------------------------------------------------------------------
# Full ensemble output
# -----------------------------------------------------------------------------
def overall_sentiment_detail(text: str) -> Tuple[str, float, str]:
    """
    Compute full sentiment result including:
        - label
        - fused score
        - engine identifier

    Returns
    -------
    (label, probability, "xf_snow_ensemble")
    """
    p_xf = _xf_prob(text)
    p_snow = _snownlp_prob(text)
    p = W_XF * p_xf + W_SNOW * p_snow
    return _label_from_prob(p), float(p), "xf_snow_ensemble"

def overall_sentiment(text: str) -> str:
    """
    Convenience wrapper returning only the sentiment label.
    """
    return overall_sentiment_detail(text)[0]

# -----------------------------------------------------------------------------
# Clause splitting: multi-punctuation aware
# -----------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"[。！？!?；;，,]|(?<!\w)\.(?!\w)|\n|\bbut\b|但是|但")

def _split_clauses(text: str) -> List[str]:
    """
    Segment text into clauses using punctuation suitable for both English
    and Chinese. This enables localized sentiment extraction around aspect
    keywords.

    Returns
    -------
    List[str]
        List of non-empty trimmed clauses.
    """
    return [p.strip() for p in _SPLIT_RE.split(text or "") if p and p.strip()]

# -----------------------------------------------------------------------------
# Aspect-window sentiment analysis
# -----------------------------------------------------------------------------
def aspect_sentiment(text: str, aspect_keywords: List[str]) -> Tuple[str, float, str]:
    """
    Compute sentiment restricted to clauses relevant to a given aspect.

    Algorithm
    ---------
    1. Split text into clauses.
    2. Identify clauses containing any aspect keyword.
    3. If matches exist:
          - Concatenate matched clauses → sentiment source segment
       Else:
          - Use full text
    4. Run ensemble sentiment on this segment.

    Parameters
    ----------
    text : str
        Original feedback text.
    aspect_keywords : List[str]
        Canonical keyword list for the target aspect.

    Returns
    -------
    (label, probability, engine)
        Same format as `overall_sentiment_detail`.
    """
    clauses = _split_clauses(text)
    if not clauses:
        return overall_sentiment_detail(text)

    kws = [k.strip().lower() for k in aspect_keywords or [] if k and k.strip()]
    hits = []

    for c in clauses:
        if _clause_has_keyword(c, kws):
            hits.append(c)

    # If aspect-specific clauses found, restrict context
    segment = " ".join(hits) if hits else text

    return overall_sentiment_detail(segment)
