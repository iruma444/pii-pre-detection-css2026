from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.schemas import PIIEntity, PIIType


@dataclass
class LLMOnlyStats:
    calls: int = 0
    seconds: float = 0.0
    failures: int = 0


class FullTextLlmExtractor:
    """Full-text local-LLM PII extraction baseline.

    This baseline intentionally sends every benchmark document to the same local
    model used by the proposed refiner. Returned spans are validated against the
    source text to prevent hallucinated offsets from being counted as detections.
    """

    ALLOWED_TYPES = {
        "PERSON": PIIType.PERSON,
        "EMAIL": PIIType.EMAIL,
        "PHONE": PIIType.PHONE,
        "ADDRESS": PIIType.ADDRESS,
        "AGE": PIIType.AGE,
        "DRIVER_ID": PIIType.DRIVER_ID,
        "CREDIT_CARD": PIIType.CREDIT_CARD,
        "BANK_ACCOUNT": PIIType.BANK_ACCOUNT,
        "IBAN": PIIType.IBAN,
    }

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": list(ALLOWED_TYPES.keys()),
                        },
                        "text": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                    "required": ["type", "text", "start", "end"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        api_url: str = "http://localhost:11434/api/chat",
        timeout_seconds: int = 180,
        think_level: str = "medium",
        num_ctx: int = 4096,
    ) -> None:
        if think_level not in {"low", "medium", "high"}:
            raise ValueError("think_level must be one of: low, medium, high")
        if num_ctx <= 0:
            raise ValueError("num_ctx must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.model_name = model_name
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.think_level = think_level
        self.num_ctx = num_ctx
        self.stats = LLMOnlyStats()
        self.last_response_meta: dict[str, object] = {}

    def extract(self, text: str) -> list[PIIEntity]:
        schema_json = json.dumps(self.RESPONSE_SCHEMA, ensure_ascii=False)
        prompt = f"""あなたはPII抽出器です。次の文章に実際に現れる個人識別情報だけを抽出してください。
対象タイプは PERSON, EMAIL, PHONE, ADDRESS, AGE, DRIVER_ID, CREDIT_CARD, BANK_ACCOUNT, IBAN です。

文章:
{text}

各エンティティについてtype, text, start, endを返してください。
startは0始まり、endはPythonスライスと同じく終端を含みません。
候補を推測・補完せず、文章中に存在する文字列だけを返してください。
該当するPIIがなければentitiesを空配列にしてください。

必ず次のJSON Schemaに従うJSONだけを返してください:
{schema_json}
"""
        request_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.think_level,
            # Ollama supports a JSON Schema object in `format`; this is more
            # constrained than the generic "json" mode.
            "format": self.RESPONSE_SCHEMA,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
            },
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(request_data, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        self.stats.calls += 1
        self.last_response_meta = {
            "think_level": self.think_level,
            "num_ctx": self.num_ctx,
            "timeout_seconds": self.timeout_seconds,
        }
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))

            response_text = self._response_text(raw)
            message = raw.get("message")
            thinking = message.get("thinking", "") if isinstance(message, dict) else ""
            self.last_response_meta = {
                "think_level": self.think_level,
                "num_ctx": self.num_ctx,
                "timeout_seconds": self.timeout_seconds,
                "done_reason": raw.get("done_reason"),
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
                "thinking_chars": len(thinking) if isinstance(thinking, str) else 0,
                "content_chars": len(response_text),
            }

            if not re.search(r"\{.*\}", response_text, re.DOTALL):
                raise ValueError(
                    "No JSON object in response; " + self._meta_text(self.last_response_meta)
                )
            return self._parse_entities(text, response_text)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            self.stats.failures += 1
            print(
                f"[llm-only] extraction failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return []
        finally:
            self.stats.seconds += time.perf_counter() - started

    @staticmethod
    def _meta_text(meta: dict[str, object]) -> str:
        keys = (
            "done_reason",
            "prompt_eval_count",
            "eval_count",
            "thinking_chars",
            "content_chars",
            "think_level",
            "num_ctx",
            "timeout_seconds",
        )
        return " ".join(f"{key}={meta.get(key)}" for key in keys)

    @staticmethod
    def _response_text(raw: dict) -> str:
        """Extract final assistant content from Ollama chat/generate responses."""
        message = raw.get("message")
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content
        response = raw.get("response", "")
        return response if isinstance(response, str) else ""

    def _parse_entities(self, source: str, response_text: str) -> list[PIIEntity]:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object in response")
        obj = json.loads(match.group(0))
        output: list[PIIEntity] = []
        occupied: set[tuple[int, int, PIIType]] = set()

        for item in obj.get("entities", []):
            if not isinstance(item, dict):
                continue
            pii_type = self.ALLOWED_TYPES.get(str(item.get("type", "")).upper())
            value = str(item.get("text", ""))
            if pii_type is None or not value:
                continue

            start = self._validated_start(source, value, item.get("start"), item.get("end"))
            if start is None:
                continue
            end = start + len(value)
            key = (start, end, pii_type)
            if key in occupied:
                continue
            occupied.add(key)
            output.append(PIIEntity(pii_type, value, start, end, 0.7, "llm_only"))

        return sorted(output, key=lambda e: e.start)

    @staticmethod
    def _validated_start(source: str, value: str, raw_start, raw_end) -> int | None:
        try:
            start = int(raw_start)
            end = int(raw_end)
            if 0 <= start <= end <= len(source) and source[start:end] == value:
                return start
        except (TypeError, ValueError):
            pass

        # Fall back only when the model-returned text occurs exactly once.
        occurrences = [m.start() for m in re.finditer(re.escape(value), source)]
        return occurrences[0] if len(occurrences) == 1 else None
