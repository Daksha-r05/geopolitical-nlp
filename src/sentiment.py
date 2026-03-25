"""
Sentiment analysis module.

Computes TextBlob polarity score.
"""

from __future__ import annotations


def sentiment_polarity(text: str) -> float:
    """
    Compute sentiment polarity score in range [-1, 1] (TextBlob default).
    """

    if not (text or "").strip():
        return 0.0

    try:
        from textblob import TextBlob

        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        # Fallback: neutral sentiment if TextBlob isn't available.
        return 0.0

