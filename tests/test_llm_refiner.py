from src.extractors.llm_extractor import LlmRefiner
from src.schemas import PIIEntity, PIIType


def test_prompt_tells_model_to_replace_broad_unicode_email() -> None:
    refiner = LlmRefiner()
    prompt = refiner._build_prompt(
        "担当の佐藤様の花子@example.com",
        [
            {
                "index": 0,
                "type": "EMAIL",
                "text": "担当の佐藤様の花子@example.com",
                "source": "regex_email_unicode_broad",
                "score": 0.6,
                "context": "担当の佐藤様の花子@example.com",
            }
        ],
    )
    assert "regex_email_unicode_broad" in prompt
    assert "DISCARD してはいけません" in prompt
    assert '"text":"花子@example.com"' in prompt


def test_apply_replace_trims_broad_email_to_subspan() -> None:
    refiner = LlmRefiner()
    candidate = PIIEntity(
        type=PIIType.EMAIL,
        text="担当の佐藤様の花子@example.com",
        start=10,
        end=10 + len("担当の佐藤様の花子@example.com"),
        score=0.6,
        source="regex_email_unicode_broad",
    )
    output = refiner._apply_decisions(
        [candidate],
        {0: {"index": 0, "action": "REPLACE", "text": "花子@example.com"}},
    )
    assert len(output) == 1
    assert output[0].text == "花子@example.com"
    assert output[0].source == "regex_email_unicode_broad+llm_replace"
    assert output[0].start == candidate.start + candidate.text.index("花子@example.com")
    assert output[0].end == output[0].start + len("花子@example.com")
