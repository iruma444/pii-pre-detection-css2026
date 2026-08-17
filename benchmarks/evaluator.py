from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from src.schemas import PIIEntity


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def _norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split()).replace("-", "").replace("ー", "")


def exact_match(truth: PIIEntity, pred: PIIEntity) -> bool:
    return truth.type == pred.type and truth.start == pred.start and truth.end == pred.end


def relaxed_match(truth: PIIEntity, pred: PIIEntity) -> bool:
    if truth.type != pred.type:
        return False
    if max(truth.start, pred.start) >= min(truth.end, pred.end):
        return False
    t = _norm(truth.text)
    p = _norm(pred.text)
    return bool(t and p and (t in p or p in t))


def score_document(
    truths: Iterable[PIIEntity],
    predictions: Iterable[PIIEntity],
    *,
    mode: str = "exact",
) -> Counts:
    matcher = exact_match if mode == "exact" else relaxed_match
    remaining = list(predictions)
    tp = 0
    fn = 0

    for truth in truths:
        match_idx = next((i for i, pred in enumerate(remaining) if matcher(truth, pred)), None)
        if match_idx is None:
            fn += 1
        else:
            tp += 1
            remaining.pop(match_idx)

    return Counts(tp=tp, fp=len(remaining), fn=fn)


def aggregate(document_counts: Iterable[Counts]) -> Counts:
    total = Counts()
    for counts in document_counts:
        total = total + counts
    return total


def bootstrap_f1_ci(
    document_counts: list[Counts],
    *,
    iterations: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not document_counts:
        return 0.0, 0.0

    rng = random.Random(seed)
    n = len(document_counts)
    values: list[float] = []
    for _ in range(iterations):
        sample = [document_counts[rng.randrange(n)] for _ in range(n)]
        values.append(aggregate(sample).f1)

    values.sort()
    low_index = max(0, int((alpha / 2) * iterations))
    high_index = min(iterations - 1, int((1 - alpha / 2) * iterations) - 1)
    return values[low_index], values[high_index]
