"""
CLI entrypoint for the geopolitics NLP pipeline.

Example:
  python main.py --query "geopolitical conflict sanctions" --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from src.data_collection import fetch_geopolitical_news, build_clean_text_list
from src.preprocessing import SpacyPreprocessor
from src.ner import extract_entities, load_spacy_ner_model
from src.event_extraction import TransformersEventClassifier
from src.sentiment import sentiment_polarity
from src.knowledge_graph import build_country_knowledge_graph
from src.timeline import build_timeline
from src.impact import compute_event_type_impacts

# Set NEWSAPI_API_KEY in the environment (see README). Do not commit real keys.
DEFAULT_NEWSAPI_API_KEY = "YOUR_NEWSAPI_API_KEY"


def _fmt_list(values: list[str]) -> str:
    return ", ".join(values) if values else "None"


def run_pipeline(query: str, limit: int, debug: bool = False) -> dict[str, Any]:
    api_key = os.getenv("NEWSAPI_API_KEY", DEFAULT_NEWSAPI_API_KEY)

    preprocessor = SpacyPreprocessor()
    # Use a dedicated NER model (raw text only).
    try:
        ner_nlp = load_spacy_ner_model()
    except Exception:
        ner_nlp = preprocessor.get_nlp()
        if debug:
            print(
                "WARNING: spaCy model `en_core_web_sm` not available. "
                "NER will likely return empty lists. Install with: "
                "python -m spacy download en_core_web_sm"
            )

    event_classifier = TransformersEventClassifier()

    articles = fetch_geopolitical_news(
        api_key=api_key,
        query=query,
        language="en",
        page_size=limit,
    )

    # Clean text list isn't strictly required for the downstream steps,
    # but keeping it aligns with the project spec.
    _clean_text_list = build_clean_text_list(articles)

    processed_events = []
    for article in articles:
        raw_text = (article.get("text") or "").strip()

        if debug:
            print("\n--- RAW TEXT (before NER) ---")
            print(raw_text[:800] + ("..." if len(raw_text) > 800 else ""))

        entities = extract_entities(text=raw_text, nlp=ner_nlp)

        if debug:
            print("--- ENTITIES ---")
            print("GPE:", _fmt_list(entities.get("GPE", [])))
            print("ORG:", _fmt_list(entities.get("ORG", [])))
            print("PERSON:", _fmt_list(entities.get("PERSON", [])))

        # Preprocess separately for event classification (NOT for NER).
        processed_text = preprocessor.preprocess_for_model(raw_text)

        event = event_classifier.classify_event(
            text=processed_text,
        )
        sentiment = sentiment_polarity(raw_text, debug=debug)

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
        "count": len(articles),
        "articles": processed_events,
        "timeline": timeline,
        "event_type_impacts": impacts,
        "graph": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geopolitical news NLP pipeline")
    parser.add_argument("--query", type=str, default="geopolitical conflict sanctions")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--debug", action="store_true", help="Print raw text and detected entities")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(query=args.query, limit=args.limit, debug=bool(args.debug))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        # Print a short readable summary by default.
        print("Processed articles:", result["count"])
        print("Timeline items:", len(result["timeline"]))
        print("Graph:", result["graph"])
        print("Event type impacts (top):")
        for item in result["event_type_impacts"][:5]:
            print(f"- {item['event_type']}: score={item['impact_score']:.3f} (freq={item['frequency']})")


if __name__ == "__main__":
    main()

