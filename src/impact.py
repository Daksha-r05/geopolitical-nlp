"""
Impact scoring module.

Impact Score = f(num_countries, sentiment, frequency)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compute_event_type_impacts(processed_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute impact score per event type.

    Heuristic scoring (simple but stable):
      - frequency: how often the event type appears
      - num_countries: average count of countries mentioned per event
      - sentiment: average polarity (boosts both positive and negative intensity)

    Impact Score = (avg_num_countries) * (1 + abs(avg_sentiment)) * log(1 + frequency)
    """

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in processed_events:
        event_type = (e.get("event_type") or "Neutral").strip()
        by_type[event_type].append(e)

    impacts: list[dict[str, Any]] = []
    for event_type, items in by_type.items():
        frequency = len(items)
        num_countries_list = []
        sentiment_list = []

        for item in items:
            entities = item.get("entities") or {}
            countries = entities.get("GPE") or []
            num_countries_list.append(len(set(countries)) if isinstance(countries, list) else 0)
            sentiment_list.append(_safe_float(item.get("sentiment"), 0.0))

        avg_num_countries = sum(num_countries_list) / frequency if frequency else 0.0
        avg_sentiment = sum(sentiment_list) / frequency if frequency else 0.0

        impact_score = avg_num_countries * (1.0 + abs(avg_sentiment)) * __import__("math").log(1.0 + frequency)

        impacts.append(
            {
                "event_type": event_type,
                "impact_score": float(impact_score),
                "frequency": int(frequency),
                "avg_num_countries": float(avg_num_countries),
                "avg_sentiment": float(avg_sentiment),
            }
        )

    # Sort descending by impact score.
    impacts_sorted = sorted(impacts, key=lambda x: x["impact_score"], reverse=True)
    return impacts_sorted

