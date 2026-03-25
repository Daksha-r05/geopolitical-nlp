"""
Data collection module.

Uses NewsAPI to fetch geopolitical news articles and returns a clean list
of articles with text fields for downstream NLP steps.
"""

from __future__ import annotations

from typing import Any


def _sample_articles() -> list[dict[str, Any]]:
    """Small built-in dataset so the project runs without a real API key."""

    return [
        {
            "title": "Sanctions expand as diplomatic talks continue",
            "url": "https://example.com/sanctions-diplomacy",
            "publishedAt": "2026-03-20",
            "description": "Officials announced new sanctions amid ongoing diplomacy between countries.",
            "text": "Officials announced new sanctions amid ongoing diplomacy between the United States and Russia. "
            "Leaders said talks will continue while economic measures are expanded.",
        },
        {
            "title": "Trade talks aim to stabilize regional supply chains",
            "url": "https://example.com/trade-talks",
            "publishedAt": "2026-03-18",
            "description": "Negotiators met to discuss tariffs and supply chain resilience.",
            "text": "Negotiators met to discuss tariffs and supply chain resilience involving Germany and China. "
            "Business groups called for reduced trade barriers and clearer agreements.",
        },
        {
            "title": "Conflict escalates near borders after ceasefire breaks",
            "url": "https://example.com/conflict-escalation",
            "publishedAt": "2026-03-15",
            "description": "Fighting resumed after a ceasefire collapsed; humanitarian access remains a concern.",
            "text": "Fighting resumed near the border after a ceasefire broke down. "
            "Humanitarian access remains a concern and international organizations urged restraint in Syria.",
        },
    ]


def fetch_geopolitical_news(
    api_key: str,
    query: str,
    language: str = "en",
    page_size: int = 10,
) -> list[dict[str, Any]]:
    """
    Fetch geopolitical news articles from NewsAPI.

    If the API key is missing/placeholder or the request fails, falls back to
    a small built-in sample dataset.

    Returns:
        List of article dicts with at least: title, url, publishedAt, text.
    """

    placeholder_keys = {"", "YOUR_NEWSAPI_KEY", "YOUR_API_KEY", None}
    if api_key in placeholder_keys:
        return _sample_articles()[: max(1, page_size)]

    try:
        try:
            import requests  # type: ignore
        except Exception:
            return _sample_articles()[: max(1, page_size)]

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "apiKey": api_key,
        }

        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            # Fallback to sample to keep the project running.
            return _sample_articles()[: max(1, page_size)]

        payload = resp.json()
        articles = payload.get("articles") or []
        cleaned: list[dict[str, Any]] = []

        for a in articles:
            title = a.get("title") or ""
            description = a.get("description") or ""
            published_at = a.get("publishedAt") or ""
            article_url = a.get("url") or ""
            content = a.get("content") or ""

            # Choose the best available text source.
            text = content.strip() if content.strip() else (description.strip() if description.strip() else title.strip())

            cleaned.append(
                {
                    "title": title,
                    "url": article_url,
                    "publishedAt": published_at,
                    "description": description,
                    "text": text,
                }
            )

        # If API returns no results, keep running via sample.
        if not cleaned:
            return _sample_articles()[: max(1, page_size)]

        return cleaned[:page_size]
    except Exception:
        return _sample_articles()[: max(1, page_size)]


def build_clean_text_list(articles: list[dict[str, Any]]) -> list[str]:
    """
    Return a clean list of article texts for downstream processing.
    """

    texts: list[str] = []
    for a in articles:
        text = (a.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts

