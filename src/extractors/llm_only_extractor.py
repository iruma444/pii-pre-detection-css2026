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

    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        api_url: str = "http://localhost:11434/api/chat",
        timeout_seconds: int = 180,
    ) -> None:
        self.model_name = model_name
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.stats = LLMOnlyStats()

    def extract(self, text: str) -> list[PIIEntity]:
        prompt = f"""あなたはPII抽出器です。次の文章に実際に現れる個人識別情報だけを抽出してください。
対象タイプは PERSON, EMAIL, PHONE, ADDRESS, AGE, DRIVER_ID, CREDIT_CARD, BANK_ACCOUNT, IBAN です。

文章:
{text}

出力はJSONのみとし、各エンティティについてtype, text, start, endを返してください。
startは0始まり、endはPythonスライスと同じく終端を含みません。
候補を推測・補完せず、文章中に存在する文字列だけを返してください。

{{"entities":[{{"type":"PERSON","text":"田中太郎","start":0,"end":4}}]}}
"""
        request_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(request_data, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        self.stats.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
            return self._parse_entities(text, self._response_text(raw))
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
