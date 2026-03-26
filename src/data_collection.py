
from __future__ import annotations

from typing import Any


def _strip_newsapi_truncation(text: str) -> str:
    """
    NewsAPI `content` sometimes ends with a truncation marker like:
    "... [+1234 chars]". This is not useful for NLP, so we strip it.
    """

    t = (text or "").strip()
    if "[+" in t and t.endswith("chars]"):
        t = t.split("[+", 1)[0].rstrip()
    return t


def build_article_text(title: str, description: str, content: str) -> str:
    """
    Build a meaningful raw text string for NER and other NLP steps.

    - Prefer `content` if present (after stripping truncation).
    - Else use `title + description` if available.
    - Else fall back to whichever is present.
    """

    title = (title or "").strip()
    description = (description or "").strip()
    content = _strip_newsapi_truncation(content or "")

    if content:
        return content
    if title and description:
        return f"{title} {description}".strip()
    return title or description


def _sample_articles() -> list[dict[str, Any]]:


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

            text = build_article_text(title=title, description=description, content=content)

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
   

    texts: list[str] = []
    for a in articles:
        text = (a.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts

