"""
Named Entity Recognition module.

Extracts:
- Countries (GPE)
- Organizations (ORG)
- Persons (PERSON)
"""

from __future__ import annotations

from typing import Any, Optional


def load_spacy_ner_model() -> Any:
    """
    Load spaCy's English NER model.

    Note:
      This expects `en_core_web_sm` to be installed:
        python -m spacy download en_core_web_sm
    """

    import spacy  # type: ignore

    return spacy.load("en_core_web_sm")


def extract_entities(text: str, nlp: Optional[Any] = None) -> dict[str, list[str]]:
    """
    Extract named entities from the input text.

    Args:
        text: Raw input text.
        nlp: spaCy language pipeline (optional). If not provided, this function
            will try to load `en_core_web_sm`.

    Returns:
        Dict with keys: GPE, ORG, PERSON (values are deduplicated lists).
    """

    if not (text or "").strip():
        return {"GPE": [], "ORG": [], "PERSON": []}

    if nlp is None:
        try:
            nlp = load_spacy_ner_model()
        except Exception:
            # Keep pipeline runnable even if spaCy model isn't installed.
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

