from src.pipeline import PiiPipeline
from src.schemas import PipelineConfig, PIIType


def test_regex_only_detects_email_without_llm():
    pipeline = PiiPipeline(
        PipelineConfig(use_regex=True, use_dict=False, use_nlp=False, use_llm=False)
    )
    entities, masked = pipeline.process("連絡先はtaro@example.comです。")
    assert any(e.type == PIIType.EMAIL and e.text == "taro@example.com" for e in entities)
    assert "[EMAIL]" in masked
    assert pipeline.stats.llm_calls == 0


def test_regex_email_does_not_swallow_ascii_sentence_period():
    pipeline = PiiPipeline(
        PipelineConfig(use_regex=True, use_dict=False, use_nlp=False, use_llm=False)
    )
    entities, _ = pipeline.process("Contact: taro@example.com.")
    emails = [e for e in entities if e.type == PIIType.EMAIL]
    assert len(emails) == 1
    assert emails[0].text == "taro@example.com"
    assert emails[0].start == len("Contact: ")
    assert emails[0].end == len("Contact: taro@example.com")


def test_regex_only_does_not_invoke_optional_components():
    pipeline = PiiPipeline(
        PipelineConfig(use_regex=True, use_dict=False, use_nlp=False, use_llm=False)
    )
    assert pipeline.dict_extractor is None
    assert pipeline.nlp_extractor is None
    assert pipeline.llm_refiner is None
