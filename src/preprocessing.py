"""
Text preprocessing module.

Uses spaCy to tokenize, lemmatize, and remove stopwords/punctuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SpacyPreprocessor:
    """
    spaCy-based preprocessing wrapper.

    Designed to be defensive: if the English model isn't installed, it falls
    back to a blank English pipeline.
    """

    lang: str = "en"

    def __post_init__(self) -> None:
        self._nlp: Any | None = None

    def get_nlp(self) -> Any:
        """Load spaCy model or fall back to a blank pipeline."""

        if self._nlp is not None:
            return self._nlp

        try:
            import spacy  # type: ignore

            # Prefer the small English model for NER.
            self._nlp = spacy.load("en_core_web_sm")
        except Exception:
            try:
                import spacy  # type: ignore

                self._nlp = spacy.blank(self.lang)
            except Exception:
                # spaCy isn't installed; fall back to a None pipeline.
                self._nlp = None

        return self._nlp

    def preprocess_for_model(self, text: str) -> str:
        """
        Preprocess text into a normalized string suitable for classifiers.

        Pipeline:
        - tokenize
        - lemmatize
        - remove stopwords and punctuation
        """

        nlp = self.get_nlp()
        if not text:
            return ""

        # spaCy unavailable fallback: simple whitespace tokenization.
        if nlp is None:
            tokens = []
            for w in text.lower().split():
                w = w.strip(".,;:!?()[]{}\"'").strip()
                if w:
                    tokens.append(w)
            return " ".join(tokens)

        doc = nlp(text)
        tokens: list[str] = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            lemma = (token.lemma_ or "").strip().lower()
            if not lemma:
                continue
            tokens.append(lemma)

        return " ".join(tokens)

