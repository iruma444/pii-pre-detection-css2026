from __future__ import annotations

from src.schemas import PIIEntity, PIIType


class NlpExtractor:
    """GiNZA-based Japanese NER extractor.

    GiNZA v4+ exposes Sekine Extended Named Entity labels through
    ``Span.label_`` (for example ``Person``, ``City`` and ``Province``), not
    only the coarse OntoNotes labels such as ``PERSON`` and ``GPE``.  The CSS
    benchmark uses a deliberately small PII taxonomy, so we map only ENE
    categories that correspond directly to the Ai4Privacy labels used by the
    benchmark.

    The benchmark must record whether ja_ginza was actually available. If it is
    unavailable, this extractor returns no candidates instead of silently using
    another model.

    Only the shared ``tok2vec`` representation and ``ner`` component are enabled.
    The benchmark consumes only ``doc.ents``; dependency parsing, morphology,
    attribute rules, compound splitting and bunsetsu recognition are unrelated to
    that output and add substantial native CPU work on every document.
    """

    # Ai4Privacy adapter targets:
    #   GIVENNAME/SURNAME -> PERSON
    #   STREET/CITY/ZIPCODE/BUILDINGNUM -> ADDRESS
    #   EMAIL -> EMAIL
    #   TELEPHONENUM -> PHONE
    #
    # Keep this mapping taxonomy-driven. Do not add labels because of a specific
    # held-out example.
    ENE_LABEL_MAP = {
        "Person": PIIType.PERSON,
        "Province": PIIType.ADDRESS,
        "City": PIIType.ADDRESS,
        "County": PIIType.ADDRESS,
        "Road": PIIType.ADDRESS,
        "Postal_Address": PIIType.ADDRESS,
        "Email": PIIType.EMAIL,
        "Phone_Number": PIIType.PHONE,
    }

    # Backward compatibility for models/pipelines that expose coarse labels.
    COARSE_LABEL_MAP = {
        "PERSON": PIIType.PERSON,
        "GPE": PIIType.ADDRESS,
        "LOC": PIIType.ADDRESS,
        "ADDRESS": PIIType.ADDRESS,
        "EMAIL": PIIType.EMAIL,
        "PHONE": PIIType.PHONE,
        "PHONE_NUMBER": PIIType.PHONE,
    }

    def __init__(self, model_name: str = "ja_ginza") -> None:
        self.model_name = model_name
        self.nlp = None
        self.is_available = False
        try:
            import spacy

            # ja_ginza is a multi-task pipeline. For this extractor we only need
            # named entities, so do not execute parser/morphologizer/GiNZA's
            # downstream syntactic components. Keep tok2vec because the NER model
            # consumes the shared token representations.
            self.nlp = spacy.load(model_name, enable=["tok2vec", "ner"])
            self.is_available = True
        except Exception:
            self.nlp = None
            self.is_available = False

    @classmethod
    def map_label(cls, label: str) -> PIIType | None:
        """Map a GiNZA ENE/coarse label into the benchmark PII taxonomy."""
        direct = cls.ENE_LABEL_MAP.get(label)
        if direct is not None:
            return direct
        return cls.COARSE_LABEL_MAP.get(label.upper())

    def extract(self, text: str) -> list[PIIEntity]:
        if not self.is_available or self.nlp is None:
            return []

        doc = self.nlp(text)
        entities: list[PIIEntity] = []
        for ent in doc.ents:
            pii_type = self.map_label(ent.label_)
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
