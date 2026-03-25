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
from src.ner import extract_entities
from src.event_extraction import TransformersEventClassifier
from src.sentiment import sentiment_polarity
from src.knowledge_graph import build_country_knowledge_graph
from src.timeline import build_timeline
from src.impact import compute_event_type_impacts


def run_pipeline(query: str, limit: int) -> dict[str, Any]:
    api_key = os.getenv("NEWSAPI_API_KEY", "YOUR_NEWSAPI_KEY")

    preprocessor = SpacyPreprocessor()
    nlp = preprocessor.get_nlp()

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
        text = article["text"]
        entities = extract_entities(text=text, nlp=nlp)
        processed_text = preprocessor.preprocess_for_model(text)

        event = event_classifier.classify_event(
            text=processed_text,
        )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(query=args.query, limit=args.limit)

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

