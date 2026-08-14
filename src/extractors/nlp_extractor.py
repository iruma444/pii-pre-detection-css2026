from __future__ import annotations

from src.schemas import PIIEntity, PIIType


class NlpExtractor:
    """GiNZA-based Japanese NER extractor.

    The benchmark must record whether ja_ginza was actually available. If it is
    unavailable, this extractor returns no candidates instead of silently using
    another model.
    """

    LABEL_MAP = {
        "PERSON": PIIType.PERSON,
        "GPE": PIIType.ADDRESS,
        "LOC": PIIType.ADDRESS,
        "FAC": PIIType.ADDRESS,
        "ADDRESS": PIIType.ADDRESS,
    }

    def __init__(self, model_name: str = "ja_ginza") -> None:
        self.model_name = model_name
        self.nlp = None
        self.is_available = False
        try:
            import spacy

            self.nlp = spacy.load(model_name)
            self.is_available = True
        except Exception:
            self.nlp = None
            self.is_available = False

    def extract(self, text: str) -> list[PIIEntity]:
        if not self.is_available or self.nlp is None:
            return []

        doc = self.nlp(text)
        entities: list[PIIEntity] = []
        for ent in doc.ents:
            pii_type = self.LABEL_MAP.get(ent.label_.upper())
            if pii_type is None:
                continue
            entities.append(
                PIIEntity(
                    type=pii_type,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=0.7,
                    source=f"ginza:{ent.label_}",
                )
            )
        return entities
