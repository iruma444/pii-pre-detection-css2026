from src.extractors.llm_only_extractor import FullTextLlmExtractor
from src.schemas import PIIType


def test_response_text_prefers_chat_message_content():
    raw = {
        "message": {"role": "assistant", "content": '{"entities":[]}'},
        "response": "legacy",
    }
    assert FullTextLlmExtractor._response_text(raw) == '{"entities":[]}'


def test_parse_entities_accepts_valid_exact_span():
    extractor = FullTextLlmExtractor()
    source = "連絡先は田中太郎です。"
    response = '{"entities":[{"type":"PERSON","text":"田中太郎","start":4,"end":8}]}'
    entities = extractor._parse_entities(source, response)
    assert len(entities) == 1
    assert entities[0].type == PIIType.PERSON
    assert entities[0].text == "田中太郎"
    assert entities[0].start == 4
    assert entities[0].end == 8


def test_parse_entities_recovers_unique_text_when_offsets_are_wrong():
    extractor = FullTextLlmExtractor()
    source = "連絡先は田中太郎です。"
    response = '{"entities":[{"type":"PERSON","text":"田中太郎","start":0,"end":4}]}'
    entities = extractor._parse_entities(source, response)
    assert len(entities) == 1
    assert entities[0].start == 4
    assert entities[0].end == 8
