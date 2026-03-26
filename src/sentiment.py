

from __future__ import annotations

import re


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _lexicon_fallback_polarity(text: str) -> float:
    """
    Tiny fallback sentiment scorer (range roughly [-1, 1]).

    This keeps the project working even if TextBlob's sentiment backend
    isn't available in the environment.
    """

    t = (text or "").lower()
    words = re.findall(r"[a-z']+", t)
    if not words:
        return 0.0

    positive = {
        "agree",
        "agreement",
        "calm",
        "ceasefire",
        "cooperate",
        "cooperation",
        "de-escalate",
        "diplomatic",
        "improve",
        "peace",
        "progress",
        "relief",
        "stabilize",
        "support",
        "success",
    }
    negative = {
        "attack",
        "battle",
        "bomb",
        "collapse",
        "conflict",
        "crisis",
        "dead",
        "escalate",
        "fight",
        "fighting",
        "kill",
        "sanction",
        "sanctions",
        "threat",
        "violence",
        "war",
    }

    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    score = (pos - neg) / max(5, (pos + neg))
    return _clamp(float(score))


def sentiment_polarity(text: str, debug: bool = False) -> float:
    if not (text or "").strip():
        return 0.0

    try:
        from textblob import TextBlob

        polarity = float(TextBlob(text).sentiment.polarity)
        # If TextBlob returns 0 for everything in your environment, this
        # fallback still provides a useful signal.
        if polarity == 0.0:
            return _lexicon_fallback_polarity(text)
        return _clamp(polarity)
    except Exception as e:
        if debug:
            print("Sentiment: TextBlob failed, using fallback. Error:", repr(e))
        return _lexicon_fallback_polarity(text)

