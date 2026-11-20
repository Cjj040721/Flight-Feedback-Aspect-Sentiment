# -----------------------------------------------------------------------------
# aspects.py
# Laboratory: Lab1, NLP Coursework
# Date: 2025-11-19
#
# This module defines aspect categories and provides robust aspect detection
# using a hybrid approach that combines (1) lexicon-based matching and
# (2) multilingual zero-shot classification. The system is optimized for
# multilingual feedback (EN + Chinese), CJK-safe keyword detection, and
# macOS/ARM stability. Zero-shot loading is thread-safe and performed lazily.
# -----------------------------------------------------------------------------

from typing import List, Dict
import os
import re
from threading import Lock

# -----------------------------------------------------------------------------
# Runtime environment configuration
# -----------------------------------------------------------------------------
# Disable safetensors fast-path due to possible mmap SIGBUS instability on
# certain macOS/ARM environments when using large HuggingFace models.
os.environ.setdefault("SAFETENSORS_FAST", "0")

# Allow overriding of the default zero-shot model via an environment variable.
DEFAULT_ZS_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ZS_MODEL_NAME = os.environ.get("ZERO_SHOT_MODEL", DEFAULT_ZS_MODEL)

# -----------------------------------------------------------------------------
# Canonical aspect codes and display names
# -----------------------------------------------------------------------------
# ASPECTS define the controlled vocabulary for aspect categories used throughout
# the system. These codes are used in model training, prediction, storage, and UI.
ASPECTS = [
    "BOOKING_FLOW", "SITE_SPEED", "CUSTOMER_SERVICE",
    "PRICE_TRANSPARENCY", "FLIGHT_OPTIONS", "PAYMENT",
    "REFUND_CHANGE", "OTHER"
]

# Human-readable display names (not used in logic; UI only).
ASPECT_DISPLAY = {
    "BOOKING_FLOW": "订票流程 / Booking flow",
    "SITE_SPEED": "网站速度 / Site speed",
    "CUSTOMER_SERVICE": "客服 / Customer service",
    "PRICE_TRANSPARENCY": "价格透明度 / Price transparency",
    "FLIGHT_OPTIONS": "航班选择 / Flight options",
    "PAYMENT": "支付 / Payment",
    "REFUND_CHANGE": "退款/改签 / Refund/Change",
    "OTHER": "其他 / Other"
}

# -----------------------------------------------------------------------------
# Lexicon-based keywords (English + Chinese)
# -----------------------------------------------------------------------------
# These keyword lists are used for (a) lexicon-only fallback detection and
# (b) providing bonuses to zero-shot scores. They cover common expressions
# occurring in flight-booking feedback datasets, including synonyms, UI terms,
# and phrasing variations in both English and Simplified Chinese.
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "BOOKING_FLOW": [
        "book","booking","search","filter","checkout","process","steps","interface","ui","ux","form",
        "passenger","add-on","insurance","seat selection","baggage selection",
        "购票","预订","下单","流程","步骤","界面","填写","筛选","跳转","卡在下一步"
    ],
    "SITE_SPEED": [
        "slow","lag","loading","timeout","performance","responsive","hang",
        "卡顿","加载","慢","延迟","超时","速度","响应","页面卡","转圈",
        "fast", "slow", "website", "loading"
    ],
    "CUSTOMER_SERVICE": [
        "agent","support","help","ticket","call center","chat","service","customer service",
        "客服","工单","人工","服务","投诉","电话","打不通","排队","不在线","不回复"
    ],
    "PRICE_TRANSPARENCY": [
        "price","ticket price","expensive","costly","cheaper","higher","lower",
        "hidden fee","tax","surcharge","price breakdown","fare rules","extra charge",
        "expected","unexpected","than expected","more than",
        "price", "higher", "unexpected", "much higher",
        "附加费","隐藏费用","税费","价格明细","收费","不透明","规则说明","到支付页才看到",
        "价格","票价","贵","便宜","比预期"
    ],
    "FLIGHT_OPTIONS": [
        "routes","connections","layover","stopover","cabin","availability",
        "航班","航线","联程","转机","舱位","可用性","班次","机型","选择少"
    ],
    "PAYMENT": [
        "pay","payment","credit card","debit","alipay","wechat","charge","refund hold",
        "payment", "pay", "failed", "declined", "charged",
        "支付","付款","支付失败","扣款","通道","银行卡","信用卡","无法支付"
    ],
    "REFUND_CHANGE": [
        "refund","chargeback","change","rebook","reschedule","cancel",
        "change my flight", "change flight", "reschedule", "change", "refund",
        "退票","退款","改签","变更","退改","手续费","进度","退款慢"
    ],
    "OTHER": []
}

# Zero-shot candidate labels: mixed EN + Chinese strings empirically yield
# more stable performance in multilingual NLI models.
ZS_LABELS: Dict[str, str] = {
    "BOOKING_FLOW": "关于订票流程 / Booking flow",
    "SITE_SPEED": "关于网站速度 / Site speed",
    "CUSTOMER_SERVICE": "关于客服 / Customer service",
    "PRICE_TRANSPARENCY": "关于价格透明度 / Price transparency",
    "FLIGHT_OPTIONS": "关于航班选择 / Flight options",
    "PAYMENT": "关于支付 / Payment",
    "REFUND_CHANGE": "关于退款或改签 / Refund or change",
    "OTHER": "其他 / Other"
}

# -----------------------------------------------------------------------------
# Utilities: CJK-safe keyword matching
# -----------------------------------------------------------------------------
def _has_cjk(s: str) -> bool:
    """
    Return True if the string contains CJK Unified Ideographs.

    Notes
    -----
    CJK text does not use whitespace segmentation, so standard regex word-boundary
    rules (\\b) do not apply. This function is used to decide how tokens should
    be matched inside text.
    """
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


def _token_in_text(token: str, text: str) -> bool:
    """
    Robust keyword matching function designed for mixed English–Chinese text.

    Rules
    -----
    - Chinese tokens: direct substring match (no whitespace segmentation).
    - English tokens:
        * lowercase substring match for multi-word tokens,
        * fallback to boundary-aware regex matching for single words.

    Returns
    -------
    bool
        True if the token is detected in the text.
    """
    if not token or not text:
        return False

    if _has_cjk(token):
        return token in text

    t = text.lower()
    tok = token.lower().strip()

    # Multi-word English tokens (e.g., "credit card")
    if " " in tok:
        return tok in t

    # Single English words use boundaries to avoid matches such as "fast" in "breakfast".
    return re.search(r"\b" + re.escape(tok) + r"\b", t) is not None


def detect_aspects_lexicon(text: str) -> List[str]:
    """
    Perform lexicon-only aspect detection.

    Parameters
    ----------
    text : str
        Input feedback text.

    Returns
    -------
    List[str]
        A list of matched aspect codes. If no aspects match, returns ["OTHER"].

    Notes
    -----
    This function acts as both a fallback and a complement to zero-shot detection.
    """
    t = text or ""
    hits: List[str] = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        if aspect == "OTHER":
            continue
        if any(_token_in_text(kw, t) for kw in keywords):
            hits.append(aspect)

    return hits or ["OTHER"]

# -----------------------------------------------------------------------------
# Zero-shot pipeline (thread-safe lazy loading)
# -----------------------------------------------------------------------------
_ZS = None
_ZS_LOCK = Lock()

def _get_zero_shot():
    """
    Load and return a HuggingFace zero-shot-classification pipeline.

    Thread-safety
    -------------
    Uses a global mutex to ensure that model initialization occurs exactly once
    (lazy loading). If loading fails, a sentinel value (False) is stored so that
    repeated attempts do not re-trigger heavy initialization.

    Returns
    -------
    pipeline or None
        Loaded zero-shot pipeline, or None if loading failed.

    Notes
    -----
    - The model is forced to CPU mode for maximal stability on macOS/MPS.
    - The default model mDeBERTa v3 XNLI has strong multilingual NLI performance.
    """
    global _ZS
    if _ZS is not None:
        return _ZS if _ZS else None

    with _ZS_LOCK:
        if _ZS is not None:
            return _ZS if _ZS else None
        try:
            from transformers import pipeline
            _ZS = pipeline(
                task="zero-shot-classification",
                model=ZS_MODEL_NAME,
                device=-1,
                truncation=True
            )
            return _ZS
        except Exception:
            _ZS = False
            return None

# -----------------------------------------------------------------------------
# Hybrid aspect detection using zero-shot + lexicon + fallback mechanisms
# -----------------------------------------------------------------------------
DEFAULT_ZS_THRESHOLD = 0.35   # Lower threshold increases recall.
LEXICON_BONUS = 0.20          # Score bonus when keywords are matched.
TOP_SCORE_MARGIN = 0.03       # Keep only candidates close to the best NLI score.

def detect_aspects(
    text: str,
    use_zero_shot: bool = True,
    threshold: float = DEFAULT_ZS_THRESHOLD,
    topk_fallback: int = 3
) -> List[str]:
    """
    Detect aspect categories using a multi-stage hybrid strategy.

    Detection Strategy
    ------------------
    1. Zero-shot NLI (multi-label):
       - Compute NLI relevance scores for each aspect label.
       - Apply lexicon bonus (+0.20) when keywords also match.
       - Accept high-scoring candidates within 0.03 of the best score to
         suppress unrelated labels that all receive high absolute scores.

    2. Zero-shot fallback:
       - If no aspect passes the threshold, select top-K aspects with
         minimal score filtering (s > 0.05) to avoid collapsing to OTHER
         in noisy or short texts.

    3. Lexicon combination:
       - Compute lexicon-only results.
       - Combine zero-shot and lexicon sets (union minus duplicates).
       - If resulting set is empty, return ["OTHER"].

    Parameters
    ----------
    text : str
        Input feedback text.
    use_zero_shot : bool
        Whether zero-shot inference is enabled.
    threshold : float
        Minimum score for accepting an aspect in step (1).
    topk_fallback : int
        Number of fallback aspects to return if thresholding yields none.

    Returns
    -------
    List[str]
        A list of selected aspect codes.

    Notes
    -----
    - The hybrid approach improves robustness across languages and feedback types.
    - The lexicon bonus is especially helpful for short inputs.
    """
    selected: List[str] = []
    zs_scores: Dict[str, float] = {}

    if use_zero_shot:
        zs = _get_zero_shot()
        if zs:
            try:
                candidates = [ZS_LABELS[a] for a in ASPECTS if a != "OTHER"]
                out = zs(sequences=text, candidate_labels=candidates, multi_label=True)
                label2score = dict(zip(out["labels"], out["scores"]))

                # Build score table with lexicon bonus
                for code in ASPECTS:
                    if code == "OTHER":
                        continue
                    sc = float(label2score.get(ZS_LABELS[code], 0.0))

                    # Apply lexicon bonus when keywords match
                    if any(_token_in_text(kw, text or "") for kw in ASPECT_KEYWORDS[code]):
                        sc = min(1.0, sc + LEXICON_BONUS)

                    zs_scores[code] = sc

                # Multi-label NLI can assign high scores to unrelated candidates.
                # Keep candidates near the best score, then add explicit lexicon hits.
                if zs_scores:
                    best_score = max(zs_scores.values())
                    selected = [
                        code for code, score in zs_scores.items()
                        if score >= threshold and best_score - score <= TOP_SCORE_MARGIN
                    ]

                # Zero-shot fallback: top-K by score if none passed threshold
                if not selected and zs_scores:
                    top = sorted(zs_scores.items(), key=lambda kv: kv[1], reverse=True)[:topk_fallback]
                    selected = [a for a, s in top if s > 0.05]

            except Exception:
                selected = []

    # Lexicon matches complement NLI predictions. If the model is unavailable,
    # detect_aspects_lexicon still provides a deterministic offline fallback.
    lex = detect_aspects_lexicon(text or "")
    if "OTHER" in lex and selected:
        lex = [a for a in lex if a != "OTHER"]

    merged = selected + [a for a in lex if a not in selected]
    return merged or ["OTHER"]
