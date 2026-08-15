from __future__ import annotations

import time
import unicodedata
from dataclasses import replace

from src.extractors.dict_extractor import DictExtractor
from src.extractors.llm_extractor import LlmRefiner
from src.extractors.nlp_extractor import NlpExtractor
from src.extractors.regex_extractor import RegexExtractor
from src.schemas import PIIEntity, PipelineConfig, RuntimeStats


class PiiPipeline:
    """Ablation-ready PII detection pipeline used in the CSS 2026 experiments."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.regex_extractor = RegexExtractor() if self.config.use_regex else None
        self.dict_extractor = DictExtractor() if self.config.use_dict else None
        self.nlp_extractor = NlpExtractor() if self.config.use_nlp else None
        self.llm_refiner = (
            LlmRefiner(
                self.config.llm_model,
                timeout_seconds=self.config.llm_timeout_seconds,
                think_level=self.config.llm_think_level,
                num_ctx=self.config.llm_num_ctx,
            )
            if self.config.use_llm
            else None
        )
        self.stats = RuntimeStats()

    @property
    def ginza_available(self) -> bool | None:
        return None if self.nlp_extractor is None else self.nlp_extractor.is_available

    def reset_stats(self) -> None:
        self.stats = RuntimeStats()
        if self.llm_refiner is not None:
            self.llm_refiner.stats.calls = 0
            self.llm_refiner.stats.seconds = 0.0
            self.llm_refiner.stats.failures = 0

    def process(self, text: str) -> tuple[list[PIIEntity], str]:
        started = time.perf_counter()
        self.stats.documents += 1

        # NFKC is used only for extractor-friendly matching when offsets are not
        # changed. If normalization changes string length, we retain the original
        # string so that benchmark spans remain valid.
        normalized = unicodedata.normalize("NFKC", text)
        working_text = normalized if len(normalized) == len(text) else text

        candidates: list[PIIEntity] = []
        if self.regex_extractor is not None:
            candidates.extend(self.regex_extractor.extract(working_text))
        if self.dict_extractor is not None:
            candidates.extend(self.dict_extractor.extract(working_text))
        if self.nlp_extractor is not None:
            candidates.extend(self.nlp_extractor.extract(working_text))

        candidates = self._deduplicate(candidates)
        candidates = self._deterministic_boundary_split(candidates)

        before_llm_calls = self.llm_refiner.stats.calls if self.llm_refiner else 0
        before_llm_seconds = self.llm_refiner.stats.seconds if self.llm_refiner else 0.0
        if self.llm_refiner is not None:
            candidates = self.llm_refiner.refine(working_text, candidates)
        after_llm_calls = self.llm_refiner.stats.calls if self.llm_refiner else 0
        after_llm_seconds = self.llm_refiner.stats.seconds if self.llm_refiner else 0.0

        final_entities = self._resolve_overlaps(candidates)
        masked = self._mask(text, final_entities)

        self.stats.total_seconds += time.perf_counter() - started
        self.stats.llm_calls += after_llm_calls - before_llm_calls
        self.stats.llm_seconds += after_llm_seconds - before_llm_seconds
        return final_entities, masked

    @staticmethod
    def _deduplicate(candidates: list[PIIEntity]) -> list[PIIEntity]:
        best: dict[tuple, PIIEntity] = {}
        for c in candidates:
            key = (c.type, c.start, c.end)
            previous = best.get(key)
            if previous is None or c.score > previous.score:
                best[key] = c
        return sorted(best.values(), key=lambda x: (x.start, x.end, x.type.value))

    @staticmethod
    def _deterministic_boundary_split(candidates: list[PIIEntity]) -> list[PIIEntity]:
        """Split clearly delimited PERSON candidates before LLM refinement.

        This stage handles only deterministic delimiters. Ambiguous boundaries
        are intentionally left to the local LLM stage.
        """
        separators = ("、", ",", "，")
        honorifics = ("様", "さん", "氏", "殿")
        output: list[PIIEntity] = []
        for c in candidates:
            if c.type.value != "PERSON" or not any(sep in c.text for sep in separators):
                output.append(c)
                continue

            cursor = 0
            pieces: list[PIIEntity] = []
            for sep in separators:
                if sep not in c.text:
                    continue
                raw_parts = c.text.split(sep)
                pieces = []
                cursor = 0
                for raw in raw_parts:
                    stripped = raw.strip()
                    for suffix in honorifics:
                        if stripped.endswith(suffix):
                            stripped = stripped[: -len(suffix)].rstrip()
                            break
                    if not stripped:
                        cursor += len(raw) + len(sep)
                        continue
                    rel = c.text.find(stripped, cursor)
                    if rel < 0:
                        pieces = []
                        break
                    start = c.start + rel
                    pieces.append(
                        replace(
                            c,
                            text=stripped,
                            start=start,
                            end=start + len(stripped),
                            source=f"{c.source}+rule_split",
                        )
                    )
                    cursor = rel + len(stripped) + len(sep)
                break
            output.extend(pieces if len(pieces) >= 2 else [c])
        return output

    @staticmethod
    def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
        """Resolve overlaps deterministically by score, then span length."""
        ranked = sorted(
            entities,
            key=lambda e: (e.score, e.end - e.start),
            reverse=True,
        )
        kept: list[PIIEntity] = []
        for current in ranked:
            overlap = any(max(current.start, x.start) < min(current.end, x.end) for x in kept)
            if not overlap:
                kept.append(current)
        return sorted(kept, key=lambda e: e.start)

    @staticmethod
    def _mask(text: str, entities: list[PIIEntity]) -> str:
        masked = text
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            masked = masked[: entity.start] + f"[{entity.type.value}]" + masked[entity.end :]
        return masked
