from benchmarks.evaluator import score_document
from src.schemas import PIIEntity, PIIType


def ent(text: str, start: int, end: int) -> PIIEntity:
    return PIIEntity(PIIType.PERSON, text, start, end, 1.0, "test")


def test_exact_match_requires_boundary_equality():
    truth = [ent("田中太郎", 0, 4)]
    pred = [ent("田中", 0, 2)]
    counts = score_document(truth, pred, mode="exact")
    assert (counts.tp, counts.fp, counts.fn) == (0, 1, 1)


def test_relaxed_match_accepts_contained_boundary():
    truth = [ent("田中太郎", 0, 4)]
    pred = [ent("田中", 0, 2)]
    counts = score_document(truth, pred, mode="relaxed")
    assert (counts.tp, counts.fp, counts.fn) == (1, 0, 0)


def test_wrong_type_is_not_match():
    truth = [ent("田中太郎", 0, 4)]
    pred = [PIIEntity(PIIType.ADDRESS, "田中太郎", 0, 4, 1.0, "test")]
    counts = score_document(truth, pred, mode="exact")
    assert (counts.tp, counts.fp, counts.fn) == (0, 1, 1)
