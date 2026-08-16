from benchmarks.diagnose_evaluation_gap import _presence_counts, _truth_has_match
from src.schemas import PIIEntity, PIIType


def ent(pii_type: PIIType, text: str, start: int, end: int) -> PIIEntity:
    return PIIEntity(pii_type, text, start, end)


def test_presence_scoring_ignores_entity_multiplicity_and_boundaries():
    truths = [
        ent(PIIType.PERSON, "田中太郎", 0, 4),
        ent(PIIType.PERSON, "鈴木花子", 5, 9),
        ent(PIIType.EMAIL, "a@example.com", 10, 23),
    ]
    predictions = [
        # Wrong PERSON span, but the type is present in the document.
        ent(PIIType.PERSON, "別人", 30, 32),
        ent(PIIType.EMAIL, "a@example.com", 10, 23),
    ]

    total, per_type = _presence_counts(
        truths,
        predictions,
        {PIIType.PERSON, PIIType.EMAIL, PIIType.PHONE},
    )

    assert total.tp == 2
    assert total.fp == 0
    assert total.fn == 0
    assert per_type[PIIType.PERSON].tp == 1
    assert per_type[PIIType.EMAIL].tp == 1
    assert per_type[PIIType.PHONE].tp == 0


def test_truth_has_match_distinguishes_exact_and_relaxed_boundary_match():
    truth = ent(PIIType.PERSON, "田中太郎", 0, 4)
    predictions = [ent(PIIType.PERSON, "田中太郎様", 0, 5)]

    assert not _truth_has_match(truth, predictions, mode="exact")
    assert _truth_has_match(truth, predictions, mode="relaxed")
