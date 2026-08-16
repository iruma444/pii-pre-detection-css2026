from __future__ import annotations

import re

from src.schemas import PIIEntity, PIIType


class RegexExtractor:
    """Deterministic baseline for structured PII.

    Rules are intentionally generic and contain no evaluation-sample-specific
    exceptions. They are kept in one place so that the exact benchmark rules
    are reproducible from the repository revision used for the paper.

    The patterns are aligned with the coarse Ai4Privacy taxonomy used by
    ``benchmarks.ai4privacy_adapter``. In particular, ZIPCODE is evaluated as
    ADDRESS, and generated EMAIL/PHONE/CREDIT_CARD values may use formatting
    that is broader than the narrow Japanese examples used in the early pilot.
    """

    PATTERNS: dict[PIIType, str] = {
        # ``\w`` is Unicode-aware in Python, so the local part can contain
        # Japanese characters. Keep the domain ASCII/punycode-style here so a
        # following Japanese particle is not accidentally absorbed into the
        # address span.
        PIIType.EMAIL: (
            r"[\w.!#$%&'*+/=?^`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
        ),
        # Accept common Japanese/international separators, including dots used
        # by some generated Ai4Privacy telephone values.
        PIIType.PHONE: (
            r"(?<![0-9０-９])"
            r"(?:(?:0|０|\+81|＋８１)[-ー−－.．\s0-9０-９]{9,18}"
            r"|[0-9０-９]{2,4}[-ー−－.．\s][0-9０-９]{2,4}"
            r"[-ー−－.．\s]?[0-9０-９]{3,4})"
            r"(?![0-9０-９])"
        ),
        # Japanese postal codes are part of ADDRESS in the benchmark adapter.
        # Require the standard 3-4 separator form to avoid treating every
        # seven-digit identifier as an address.
        PIIType.ADDRESS: (
            r"(?<![0-9０-９])(?:〒\s*)?[0-9０-９]{3}[-ー−－][0-9０-９]{4}"
            r"(?![0-9０-９])"
        ),
        # Ai4Privacy CREDITCARDNUMBER values are synthetic and can be broader
        # than the 14--16 digit range used by the early pilot. Cover the common
        # 13--19 digit PAN range while retaining the existing separated format.
        PIIType.CREDIT_CARD: (
            r"(?<![0-9０-９])(?:[0-9０-９]{13,19}"
            r"|(?:[0-9０-９]{4}[-ー−－\s]+){3}[0-9０-９]{3,4})(?![0-9０-９])"
        ),
        PIIType.BANK_ACCOUNT: (
            r"(?<![0-9０-９])(?:[0-9０-９]{7}|[0-9０-９]{4}[-－ー−\s][0-9０-９]{3})"
            r"(?![0-9０-９])"
        ),
        PIIType.IBAN: (
            r"(?<![A-Za-z0-9])[A-Z]{2}[0-9A-Z]{2}(?:[ -]?[A-Za-z0-9]){11,38}"
            r"(?![A-Za-z0-9])"
        ),
    }

    def extract(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(
                    PIIEntity(
                        type=pii_type,
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        score=0.8 if pii_type == PIIType.BANK_ACCOUNT else 1.0,
                        source="regex",
                    )
                )
        return entities
