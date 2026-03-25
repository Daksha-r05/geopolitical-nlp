"""
Streamlit dashboard for geopolitical event extraction.
"""

from __future__ import annotations

import os
from typing import Any

from src.data_collection import build_clean_text_list, fetch_geopolitical_news
from src.event_extraction import TransformersEventClassifier
from src.impact import compute_event_type_impacts
from src.knowledge_graph import build_country_knowledge_graph, draw_country_knowledge_graph
from src.ner import extract_entities
from src.preprocessing import SpacyPreprocessor
from src.sentiment import sentiment_polarity
from src.timeline import build_timeline

# Import `streamlit` independently so the app can still run even if optional
# visualization deps like matplotlib are missing.
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


if st is not None:  # pragma: no cover
    st.set_page_config(
        page_title="Geopolitical Event Extraction",
        page_icon="🌍",
        layout="wide",
    )


def _require_streamlit() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError(
            "streamlit is not installed. Install dependencies to run `streamlit run app.py`."
        )


if st is not None:  # pragma: no cover
    @st.cache_resource(show_spinner=False)
    def load_nlp() -> Any:
        return SpacyPreprocessor().get_nlp()

    @st.cache_resource(show_spinner=False)
    def load_preprocessor() -> SpacyPreprocessor:
        return SpacyPreprocessor()

    @st.cache_resource(show_spinner=False)
    def load_event_classifier() -> TransformersEventClassifier:
        return TransformersEventClassifier()

    def run_pipeline(query: str, api_key: str, limit: int) -> dict[str, Any]:
        preprocessor = load_preprocessor()
        nlp = load_nlp()
        event_classifier = load_event_classifier()

        articles = fetch_geopolitical_news(
            api_key=api_key,
            query=query,
            language="en",
            page_size=limit,
        )

        # Spec-compliance step: produce a clean text list.
        _ = build_clean_text_list(articles)

        processed_events: list[dict[str, Any]] = []
        for article in articles:
            text = article["text"]
            entities = extract_entities(text=text, nlp=nlp)
            processed_text = preprocessor.preprocess_for_model(text)

            event = event_classifier.classify_event(text=processed_text)
            sentiment = sentiment_polarity(text)

            processed_events.append(
                {
                    **article,
                    "entities": entities,
                    "event_type": event["event_type"],
                    "event_confidence": event["confidence"],
                    "sentiment": sentiment,
                }
            )

        graph = build_country_knowledge_graph(processed_events)
        timeline = build_timeline(processed_events)
        impacts = compute_event_type_impacts(processed_events)

        return {
            "query": query,
            "count": len(processed_events),
            "articles": processed_events,
            "timeline": timeline,
            "event_type_impacts": impacts,
            "graph": graph,
        }

    def sidebar() -> tuple[str, str, int]:
        st.sidebar.header("News Search")

        api_key = st.sidebar.text_input(
            "NewsAPI Key",
            value=os.getenv("NEWSAPI_API_KEY", ""),
            type="password",
            help="If empty, sample data will be used so the app can still run.",
        )

        query = st.sidebar.text_input(
            "Query",
            value="geopolitical conflict sanctions diplomacy trade",
        )

        limit = st.sidebar.slider("Articles", min_value=1, max_value=30, value=10, step=1)
        return query, api_key, limit

    def render_entities(entities: dict[str, list[str]]) -> None:
        gpe = entities.get("GPE", [])
        org = entities.get("ORG", [])
        person = entities.get("PERSON", [])

        st.markdown("**Entities**")
        st.caption(f"GPE (Countries): {', '.join(gpe) if gpe else 'None'}")
        st.caption(f"ORG (Organizations): {', '.join(org) if org else 'None'}")
        st.caption(f"PERSON (Persons): {', '.join(person) if person else 'None'}")

    def main() -> None:
        st.title("Geopolitical Event Extraction and Global Impact Analysis")
        st.write(
            "Fetch articles, extract geopolitical entities, classify events, analyze sentiment, and visualize relationships."
        )

        query, api_key, limit = sidebar()

        if st.button("Fetch and Analyze", type="primary"):
            with st.spinner("Running pipeline..."):
                result = run_pipeline(
                    query=query, api_key=api_key or "YOUR_NEWSAPI_KEY", limit=limit
                )

            articles = result["articles"]

            tab_news, tab_entities, tab_events, tab_graph, tab_timeline = st.tabs(
                ["News", "Entities", "Event Types", "Knowledge Graph", "Timeline"]
            )

            with tab_news:
                st.subheader("News Articles")
                for i, article in enumerate(articles, start=1):
                    st.markdown(f"### {i}. {article.get('title', '(untitled)')}")
                    st.caption(f"Published: {article.get('publishedAt', '')}")
                    url = article.get("url")
                    if url:
                        st.markdown(f"[Link]({url})")
                    text = article.get("text", "")
                    st.write(text[:600] + ("..." if len(text) > 600 else ""))
                    st.divider()

            with tab_entities:
                st.subheader("Extracted Entities")
                for i, article in enumerate(articles, start=1):
                    st.markdown(f"### {i}. {article.get('title', '(untitled)')}")
                    render_entities(article.get("entities", {}))
                    st.divider()

            with tab_events:
                st.subheader("Event Type & Sentiment")
                for i, article in enumerate(articles, start=1):
                    st.markdown(f"### {i}. {article.get('title', '(untitled)')}")
                    st.caption(
                        f"Event: {article.get('event_type')} (confidence={article.get('event_confidence'):.3f})"
                    )
                    st.caption(f"Sentiment polarity: {article.get('sentiment'):.3f}")
                    st.divider()

                st.subheader("Impact Scores by Event Type")
                for item in result["event_type_impacts"]:
                    st.markdown(
                        f"- **{item['event_type']}**: impact_score={item['impact_score']:.3f} (frequency={item['frequency']})"
                    )

            with tab_graph:
                st.subheader("Knowledge Graph (Countries)")
                st.caption("Nodes represent countries; edges represent co-occurrence with sentiment-weighted edges.")

                if plt is None:
                    st.error("matplotlib is not installed.")
                else:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    draw_country_knowledge_graph(result["graph"], ax=ax)
                    st.pyplot(fig)

            with tab_timeline:
                st.subheader("Timeline")
                for item in result["timeline"]:
                    date = item.get("date", "")
                    st.markdown(
                        f"**{date}** - {item.get('event_type')} | {item.get('title','')}"
                    )
                    countries = item.get("countries", [])
                    if countries:
                        st.caption("Countries: " + ", ".join(countries))
                    url = item.get("url")
                    if url:
                        st.markdown(f"[Link]({url})")
                    st.divider()

else:

    def main() -> None:  # pragma: no cover
        _require_streamlit()


if __name__ == "__main__":  # pragma: no cover
    main()

