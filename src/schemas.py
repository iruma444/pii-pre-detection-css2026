from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any


class PIIType(str, Enum):
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    AGE = "AGE"
    DRIVER_ID = "DRIVER_ID"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"


@dataclass(frozen=True)
class PIIEntity:
    type: PIIType
    text: str
    start: int
    end: int
    score: float = 1.0
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


@dataclass(frozen=True)
class PipelineConfig:
    use_regex: bool = True
    use_dict: bool = True
    use_nlp: bool = True
    use_llm: bool = True
    llm_model: str = "gpt-oss:20b"
    # None preserves Ollama/model defaults for existing benchmark entry points.
    # Durable experiments can set these explicitly for matched comparisons.
    llm_think_level: str | None = None
    llm_num_ctx: int | None = None
    llm_timeout_seconds: int = 180


@dataclass
class RuntimeStats:
    documents: int = 0
    total_seconds: float = 0.0
    llm_calls: int = 0
    llm_seconds: float = 0.0

    @property
    def llm_call_rate(self) -> float:
        return self.llm_calls / self.documents if self.documents else 0.0
