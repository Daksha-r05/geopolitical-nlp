
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
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


def _load_filter_nlp() -> Any | None:
    """
    Load spaCy model for lightweight NLP filtering.
    Falls back to None if unavailable.
    """

    try:
        import spacy  # type: ignore

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _is_geopolitical_article(text: str, nlp: Any | None) -> bool:
    """
    Lightweight NLP-based relevance filter.

    We do NOT rely only on keywords:
    - Uses entity signals (GPE/ORG/PERSON) when spaCy model is available
    - Uses keyword hints as a secondary signal only
    """

    t = (text or "").strip()
    if not t:
        return False

    keyword_score = 0
    keyword_patterns = [
        r"\bdiplomac\w*",
        r"\bgeopolitic\w*",
        r"\binternational\b",
        r"\bforeign policy\b",
        r"\bsanction\w*",
        r"\bconflict\w*",
        r"\btrade\b",
        r"\btariff\w*",
        r"\bceasefire\b",
        r"\bgovernment\b",
        r"\bministry\b",
        r"\bunited nations\b",
        r"\bnato\b",
        r"\beu\b",
    ]
    for p in keyword_patterns:
        if re.search(p, t, flags=re.IGNORECASE):
            keyword_score += 1

    if nlp is None:
        # Fallback only if NLP model isn't available.
        return keyword_score >= 2

    try:
        doc = nlp(t)
        gpe_count = sum(1 for e in doc.ents if e.label_ == "GPE")
        org_count = sum(1 for e in doc.ents if e.label_ == "ORG")
        person_count = sum(1 for e in doc.ents if e.label_ == "PERSON")

        # NLP-driven relevance rules:
        # - strong geo signal via multiple GPEs
        # - or one GPE + institution/person + at least one political/economic cue
        if gpe_count >= 2:
            return True
        if gpe_count >= 1 and (org_count >= 1 or person_count >= 1) and keyword_score >= 1:
            return True
        if gpe_count >= 1 and keyword_score >= 2:
            return True

        return False
    except Exception:
        return keyword_score >= 2


def _tokenize_query(query: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", (query or "").lower())
    stop = {"the", "and", "for", "with", "from", "that", "this", "into", "about", "world", "news"}
    return {t for t in tokens if t not in stop}


def _score_article_match_importance(article: dict[str, Any], query_tokens: set[str], nlp: Any | None) -> float:
    """
    Combined score for ranking:
    - query match score (primary)
    - entity/importance score (secondary)
    - recency score (small boost)
    """

    text = (article.get("text") or "").strip()
    title = (article.get("title") or "").strip()
    full_text = f"{title} {text}".lower()
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", full_text))

    # Query match
    overlap = len(query_tokens.intersection(words))
    match_score = overlap / max(1, len(query_tokens))

    # NLP importance signal (entities) if model available
    entity_score = 0.0
    if nlp is not None and text:
        try:
            doc = nlp(text)
            gpe = sum(1 for e in doc.ents if e.label_ == "GPE")
            org = sum(1 for e in doc.ents if e.label_ == "ORG")
            person = sum(1 for e in doc.ents if e.label_ == "PERSON")
            entity_score = min(1.0, (gpe * 0.35 + org * 0.20 + person * 0.10))
        except Exception:
            entity_score = 0.0

    # Recency (last 7 days gets higher weight)
    recency = 0.0
    published_at = (article.get("publishedAt") or "").strip()
    if published_at:
        try:
            import dateparser  # type: ignore

            dt = dateparser.parse(published_at)
            if dt is not None:
                now = datetime.now(timezone.utc)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
                recency = max(0.0, 1.0 - min(age_days, 7.0) / 7.0)
        except Exception:
            recency = 0.0

    # Weighted score: highest match first, then importance, then recency
    return 0.65 * match_score + 0.25 * entity_score + 0.10 * recency


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
        broad_query = f"({query}) OR international OR politics OR world"
        nlp = _load_filter_nlp()
        query_tokens = _tokenize_query(query)

        # Fetch multiple pages so we can filter and still return enough items.
        max_pages = 4
        api_page_size = min(50, max(20, page_size))
        articles: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        now_utc = datetime.now(timezone.utc)
        from_date = (now_utc - timedelta(days=7)).date().isoformat()
        to_date = now_utc.date().isoformat()

        for page in range(1, max_pages + 1):
            params = {
                "q": broad_query,
                "language": language,
                "pageSize": api_page_size,
                "page": page,
                "from": from_date,
                "to": to_date,
                "sortBy": "relevancy",
                "apiKey": api_key,
            }
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                continue

            payload = resp.json()
            batch = payload.get("articles") or []
            if not batch:
                break

            for a in batch:
                article_url = (a.get("url") or "").strip()
                if article_url and article_url in seen_urls:
                    continue
                if article_url:
                    seen_urls.add(article_url)
                articles.append(a)

            # Stop early if we have enough candidates.
            if len(articles) >= page_size * 6:
                break

        cleaned: list[dict[str, Any]] = []

        for a in articles:
            title = a.get("title") or ""
            description = a.get("description") or ""
            published_at = a.get("publishedAt") or ""
            article_url = a.get("url") or ""
            content = a.get("content") or ""

            text = build_article_text(title=title, description=description, content=content)
            if not _is_geopolitical_article(text=text, nlp=nlp):
                continue

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

        # Rank by combined match + importance + recency.
        ranked = sorted(
            cleaned,
            key=lambda a: _score_article_match_importance(a, query_tokens=query_tokens, nlp=nlp),
            reverse=True,
        )
        return ranked[:page_size]
    except Exception:
        return _sample_articles()[: max(1, page_size)]


def build_clean_text_list(articles: list[dict[str, Any]]) -> list[str]:
   

    texts: list[str] = []
    for a in articles:
        text = (a.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts

