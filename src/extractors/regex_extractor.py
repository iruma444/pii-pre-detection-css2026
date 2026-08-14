from __future__ import annotations

import re

from src.schemas import PIIEntity, PIIType


class RegexExtractor:
    """Deterministic baseline for structured PII.

    Rules are intentionally generic and contain no evaluation-sample-specific
    exceptions. They are kept in one place so that the exact benchmark rules
    are reproducible from the repository revision used for the paper.
    """

    PATTERNS: dict[PIIType, str] = {
        PIIType.EMAIL: r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+",
        PIIType.PHONE: (
            r"(?<![0-9０-９])"
            r"(?:(?:0|０|\+81|＋８１)[-ー−－\s0-9０-９]{9,18}"
            r"|[0-9０-９]{2,4}[-ー−－\s][0-9０-９]{2,4}"
            r"[-ー−－\s]?[0-9０-９]{3,4})"
            r"(?![0-9０-９])"
        ),
        PIIType.CREDIT_CARD: (
            r"(?<![0-9０-９])(?:[0-9０-９]{14,16}"
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
