from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.schemas import PIIEntity, PIIType


@dataclass
class LLMCallStats:
    calls: int = 0
    seconds: float = 0.0
    failures: int = 0


class LlmRefiner:
    """Selective local-LLM refinement through Ollama.

    Only ambiguous candidates are sent to the model. High-confidence regex
    candidates bypass the LLM. The model can KEEP, DISCARD, or SPLIT a
    candidate. This design is deliberate: the paper evaluates both accuracy
    and the fraction of documents that actually invoke the LLM.
    """

    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        api_url: str = "http://localhost:11434/api/chat",
        timeout_seconds: int = 180,
        think_level: str | None = None,
        num_ctx: int | None = None,
    ) -> None:
        if think_level is not None and think_level not in {"low", "medium", "high"}:
            raise ValueError("think_level must be one of: low, medium, high")
        if num_ctx is not None and num_ctx <= 0:
            raise ValueError("num_ctx must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.model_name = model_name
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.think_level = think_level
        self.num_ctx = num_ctx
        self.stats = LLMCallStats()

    @staticmethod
    def is_ambiguous(candidate: PIIEntity) -> bool:
        # Structured regex matches are deliberately kept out of the LLM path.
        if candidate.source == "regex" and candidate.score >= 0.9:
            return False
        return candidate.type in {PIIType.PERSON, PIIType.ADDRESS, PIIType.BANK_ACCOUNT}

    def refine(self, text: str, candidates: list[PIIEntity]) -> list[PIIEntity]:
        ambiguous_indices = [i for i, c in enumerate(candidates) if self.is_ambiguous(c)]
        if not ambiguous_indices:
            return candidates

        payload_candidates = []
        for idx in ambiguous_indices:
            cand = candidates[idx]
            left = max(0, cand.start - 30)
            right = min(len(text), cand.end + 30)
            payload_candidates.append(
                {
                    "index": idx,
                    "type": cand.type.value,
                    "text": cand.text,
                    "context": text[left:right].replace("\n", " "),
                }
            )

        prompt = self._build_prompt(text, payload_candidates)
        options: dict[str, object] = {"temperature": 0.0}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        data: dict[str, object] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": options,
        }
        if self.think_level is not None:
            data["think"] = self.think_level

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        self.stats.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
            response_text = self._response_text(raw)
            decisions = self._parse_response(response_text)
            return self._apply_decisions(candidates, decisions)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
            self.stats.failures += 1
            print(
                f"[llm] refinement failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            # Fail-open for privacy preservation: retaining a candidate is safer
            # than discarding it when the refinement model is unavailable.
            return candidates
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

    def _build_prompt(self, text: str, candidates: list[dict]) -> str:
        candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
        return f"""あなたは日本語PII検出器の補正器です。候補の新規探索は行わず、与えられた候補だけを補正してください。

対象文:
{text}

候補:
{candidates_json}

各候補について次のいずれかを判定してください。
- KEEP: 候補範囲がPIIとして妥当。
- DISCARD: PIIではない偽陽性。
- SPLIT: 1候補に複数の同種PIIが結合している。partsに候補文字列内の各PII文字列を順番に入れる。

境界補正が必要だが分割ではない場合は action=REPLACE とし、textに候補内部の正しいPII部分を入れてください。
候補外の文字列を新規に追加してはいけません。

JSONのみを返してください。
{{"decisions":[
  {{"index":0,"action":"KEEP"}},
  {{"index":1,"action":"DISCARD"}},
  {{"index":2,"action":"SPLIT","parts":["田中太一","鈴木花子"]}},
  {{"index":3,"action":"REPLACE","text":"田中太一"}}
]}}
"""

    @staticmethod
    def _parse_response(response_text: str) -> dict[int, dict]:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object in LLM response")
        obj = json.loads(match.group(0))
        decisions = obj.get("decisions", [])
        result: dict[int, dict] = {}
        for item in decisions:
            if not isinstance(item, dict) or "index" not in item:
                continue
            result[int(item["index"])] = item
        return result

    def _apply_decisions(self, candidates: list[PIIEntity], decisions: dict[int, dict]) -> list[PIIEntity]:
        output: list[PIIEntity] = []
        for idx, cand in enumerate(candidates):
            decision = decisions.get(idx)
            if decision is None:
                output.append(cand)
                continue

            action = str(decision.get("action", "KEEP")).upper()
            if action == "DISCARD":
                continue
            if action == "KEEP":
                output.append(cand)
                continue
            if action == "REPLACE":
                value = str(decision.get("text", ""))
                replacement = self._subspan(cand, value)
                output.append(replacement if replacement else cand)
                continue
            if action == "SPLIT":
                parts = decision.get("parts", [])
                split_entities = self._split_candidate(cand, parts)
                output.extend(split_entities if split_entities else [cand])
                continue

            output.append(cand)
        return output

    @staticmethod
    def _subspan(candidate: PIIEntity, value: str) -> PIIEntity | None:
        if not value:
            return None
        rel = candidate.text.find(value)
        if rel < 0:
            return None
        start = candidate.start + rel
        return PIIEntity(
            type=candidate.type,
            text=value,
            start=start,
            end=start + len(value),
            score=max(candidate.score, 0.75),
            source=f"{candidate.source}+llm_replace",
        )

    @staticmethod
    def _split_candidate(candidate: PIIEntity, parts: list) -> list[PIIEntity]:
        output: list[PIIEntity] = []
        cursor = 0
        for raw_part in parts:
            part = str(raw_part)
            if not part:
                continue
            rel = candidate.text.find(part, cursor)
            if rel < 0:
                return []
            start = candidate.start + rel
            output.append(
                PIIEntity(
                    type=candidate.type,
                    text=part,
                    start=start,
                    end=start + len(part),
                    score=max(candidate.score, 0.75),
                    source=f"{candidate.source}+llm_split",
                )
            )
            cursor = rel + len(part)
        return output
