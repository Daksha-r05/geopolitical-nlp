"""
Event extraction module.

Classifies geopolitical events into:
- Conflict
- Diplomacy
- Trade
- Sanctions
- Neutral

Primary approach:
- Transformers zero-shot classification (BART MNLI) with candidate labels.

Fallback:
- Simple keyword heuristics if transformers/model download fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EVENT_TYPES = ["Conflict", "Diplomacy", "Trade", "Sanctions", "Neutral"]


@dataclass
class TransformersEventClassifier:
    model_name: str = "facebook/bart-large-mnli"
    device: int | None = None

    def __post_init__(self) -> None:
        self._pipeline: Any | None = None
        self._use_keyword_fallback = False
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="zero-shot-classification",
                model=self.model_name,
                device=self.device if self.device is not None else -1,
                # Keep the project runnable offline (no model download).
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            # Run without transformers availability (e.g., offline env).
            self._pipeline = None
            self._use_keyword_fallback = True

    def _keyword_fallback(self, text: str) -> dict[str, float]:
        """Very lightweight classifier so the pipeline still runs."""

        t = (text or "").lower()

        scores = {k: 0.0 for k in EVENT_TYPES}

        # Conflict
        if any(w in t for w in ["fighting", "conflict", "clash", "battle", "war", "bomb", "escalat", "airstrike"]):
            scores["Conflict"] += 0.9

        # Diplomacy
        if any(w in t for w in ["talks", "diplomacy", "negotiat", "summit", "agreement", "ceasefire", "diplomatic"]):
            scores["Diplomacy"] += 0.8

        # Trade
        if any(w in t for w in ["trade", "tariff", "import", "export", "agreement", "market access", "supply chain"]):
            scores["Trade"] += 0.8

        # Sanctions
        if any(w in t for w in ["sanction", "embargo", "blacklist", "restricted", "export controls", "punish"]):
            scores["Sanctions"] += 0.95

        if all(v == 0.0 for v in scores.values()):
            scores["Neutral"] = 0.6

        # Convert to a pseudo-confidence distribution.
        best_label = max(scores, key=scores.get)
        conf = scores[best_label]
        if best_label == "Neutral" and conf == 0.0:
            conf = 0.2
        return {best_label: conf}

    def classify_event(self, text: str) -> dict[str, Any]:
        """
        Classify event type from input text.

        Returns:
            {"event_type": str, "confidence": float}
        """

        if not (text or "").strip():
            return {"event_type": "Neutral", "confidence": 0.0}

        if self._pipeline is None or self._use_keyword_fallback:
            kw = self._keyword_fallback(text)
            event_type = next(iter(kw.keys()))
            return {"event_type": event_type, "confidence": float(kw[event_type])}

        # Zero-shot classification returns labels + scores.
        # We map the best label to our expected set.
        try:
            result = self._pipeline(
                text,
                candidate_labels=EVENT_TYPES,
                multi_label=False,
            )
            labels = result.get("labels") or []
            scores = result.get("scores") or []
            if not labels or not scores:
                return {"event_type": "Neutral", "confidence": 0.0}
            return {"event_type": labels[0], "confidence": float(scores[0])}
        except Exception:
            kw = self._keyword_fallback(text)
            event_type = next(iter(kw.keys()))
            return {"event_type": event_type, "confidence": float(kw[event_type])}

