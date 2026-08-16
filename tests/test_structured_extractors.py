from __future__ import annotations

from src.extractors.nlp_extractor import NlpExtractor
from src.extractors.regex_extractor import RegexExtractor
from src.schemas import PIIType


def _values(text: str, pii_type: PIIType) -> list[str]:
    return [entity.text for entity in RegexExtractor().extract(text) if entity.type == pii_type]


def test_regex_ascii_email_does_not_swallow_japanese_prefix() -> None:
    assert "taro@example.com" in _values("連絡先はtaro@example.comです。", PIIType.EMAIL)


def test_regex_supports_dot_separated_phone() -> None:
    assert "+81.90.1234.5678" in _values(
        "電話番号は+81.90.1234.5678です。", PIIType.PHONE
    )


def test_regex_maps_japanese_postal_code_to_address() -> None:
    assert "123-4567" in _values("郵便番号は123-4567です。", PIIType.ADDRESS)


def test_regex_supports_17_digit_credit_card_candidate() -> None:
    assert "12345678901234567" in _values(
        "カード候補は12345678901234567です。", PIIType.CREDIT_CARD
    )


def test_ginza_extended_labels_map_to_benchmark_taxonomy() -> None:
    assert NlpExtractor.map_label("Person") == PIIType.PERSON
    assert NlpExtractor.map_label("Province") == PIIType.ADDRESS
    assert NlpExtractor.map_label("City") == PIIType.ADDRESS
    assert NlpExtractor.map_label("County") == PIIType.ADDRESS
    assert NlpExtractor.map_label("Road") == PIIType.ADDRESS
    assert NlpExtractor.map_label("Postal_Address") == PIIType.ADDRESS
    assert NlpExtractor.map_label("Email") == PIIType.EMAIL
    assert NlpExtractor.map_label("Phone_Number") == PIIType.PHONE


def test_ginza_does_not_map_unrelated_location_like_country_to_address() -> None:
    assert NlpExtractor.map_label("Country") is None
