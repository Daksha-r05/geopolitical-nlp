"""
Timeline module.

Extracts dates for each event and stores events chronologically.
Uses dateparser for flexible date handling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        import dateparser

        dt = dateparser.parse(value)
        return dt
    except Exception:
        return None


def build_timeline(processed_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a chronological timeline from processed events.
    """

    items: list[dict[str, Any]] = []

    for event in processed_events:
        published_at = event.get("publishedAt") or ""
        dt = _parse_date(published_at)
        if dt is None:
            # Allow missing/invalid dates; we keep ordering stable later.
            date_str = ""
        else:
            date_str = dt.date().isoformat()

        entities = event.get("entities") or {}
        countries = entities.get("GPE") or []

        items.append(
            {
                "date": date_str,
                "event_type": event.get("event_type") or "Neutral",
                "title": event.get("title") or "",
                "url": event.get("url") or "",
                "countries": countries,
            }
        )

    # Sort: known dates first (desc), then unknown.
    def sort_key(x: dict[str, Any]) -> tuple[int, datetime | str]:
        if x.get("date"):
            # Parse back for consistent sorting.
            dt = _parse_date(x["date"])
            return (0, dt or datetime.min)
        return (1, "")  # Unknown dates at end

    items_sorted = sorted(items, key=sort_key)
    return items_sorted

