from __future__ import annotations

import re
import unicodedata

from src.schemas import PIIEntity, PIIType


class RegexExtractor:
    """Deterministic extractor for structured PII candidates.

    Rules are intentionally generic and contain no evaluation-sample-specific
    exceptions. They are kept in one place so that the exact benchmark rules
    are reproducible from the repository revision used for the paper.

    The stable regex baseline emits only candidates that can reasonably be used
    as final predictions without a contextual refiner.  The proposed method may
    additionally enable a recall-first Unicode EMAIL candidate.  That candidate
    is deliberately wider when Japanese prose is attached directly to an
    internationalized local part, so it is an *intermediate candidate* for LLM
    boundary refinement rather than a fair no-LLM baseline prediction.
    """

    PATTERNS: dict[PIIType, str] = {
        PIIType.EMAIL: (
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+"
            r"(?:\.[A-Za-z0-9-]+)+"
        ),
        PIIType.PHONE: (
            r"(?<![0-9０-９])"
            r"(?:(?:0|０|\+81|＋８１)[-ー−－.．\s0-9０-９]{9,18}"
            r"|[0-9０-９]{2,4}[-ー−－.．\s][0-9０-９]{2,4}"
            r"[-ー−－.．\s]?[0-9０-９]{3,4})"
            r"(?![0-9０-９])"
        ),
        PIIType.ADDRESS: (
            r"(?<![0-9０-９])(?:〒\s*)?[0-9０-９]{3}[-ー−－][0-9０-９]{4}"
            r"(?![0-9０-９])"
        ),
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

    EMAIL_DOMAIN = re.compile(r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
    EMAIL_LOCAL_PUNCT = frozenset(".!#$%&'*+/=?^_`{|}~-")

    def __init__(self, include_ambiguous_email_candidates: bool = False) -> None:
        self.include_ambiguous_email_candidates = include_ambiguous_email_candidates

    @classmethod
    def _is_unicode_email_local_char(cls, ch: str) -> bool:
        if ch in cls.EMAIL_LOCAL_PUNCT:
            return True
        category = unicodedata.category(ch)
        return bool(category) and category[0] in {"L", "M", "N"}

    @classmethod
    def _broad_unicode_email_candidates(
        cls, text: str, exact_emails: list[PIIEntity]
    ) -> list[PIIEntity]:
        output: list[PIIEntity] = []
        for domain in cls.EMAIL_DOMAIN.finditer(text):
            at = domain.start()
            if any(e.start <= at < e.end for e in exact_emails):
                continue

            left = at
            while left > 0 and cls._is_unicode_email_local_char(text[left - 1]):
                left -= 1
            if left == at:
                continue

            local = text[left:at]
            if local.isascii():
                continue

            output.append(
                PIIEntity(
                    type=PIIType.EMAIL,
                    text=text[left : domain.end()],
                    start=left,
                    end=domain.end(),
                    score=0.6,
                    source="regex_email_unicode_broad",
                )
            )
        return output

    def extract(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        exact_emails: list[PIIEntity] = []
        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                entity = PIIEntity(
                    type=pii_type,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    score=0.8 if pii_type == PIIType.BANK_ACCOUNT else 1.0,
                    source="regex",
                )
                entities.append(entity)
                if pii_type == PIIType.EMAIL:
                    exact_emails.append(entity)

        if self.include_ambiguous_email_candidates:
            entities.extend(self._broad_unicode_email_candidates(text, exact_emails))
        return entities
