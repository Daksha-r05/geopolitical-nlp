"""
Named Entity Recognition module.

Extracts:
- Countries (GPE)
- Organizations (ORG)
- Persons (PERSON)
"""

from __future__ import annotations

from typing import Any


def extract_entities(text: str, nlp: Any) -> dict[str, list[str]]:
    """
    Extract named entities from the input text.

    Args:
        text: Raw input text.
        nlp: spaCy language pipeline.

    Returns:
        Dict with keys: GPE, ORG, PERSON (values are deduplicated lists).
    """

    if not text:
        return {"GPE": [], "ORG": [], "PERSON": []}

    if nlp is None:
        return {"GPE": [], "ORG": [], "PERSON": []}

    doc = nlp(text)

    gpe: set[str] = set()
    org: set[str] = set()
    person: set[str] = set()

    for ent in doc.ents:
        label = ent.label_
        value = (ent.text or "").strip()
        if not value:
            continue

        if label == "GPE":
            gpe.add(value)
        elif label == "ORG":
            org.add(value)
        elif label == "PERSON":
            person.add(value)

    return {
        "GPE": sorted(gpe),
        "ORG": sorted(org),
        "PERSON": sorted(person),
    }

