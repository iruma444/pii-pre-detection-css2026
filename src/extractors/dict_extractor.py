from __future__ import annotations

import re

from src.schemas import PIIEntity, PIIType


class DictExtractor:
    """Generic dictionary/pattern extractor with no test-sample exceptions."""

    PREFECTURES = [
        "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
        "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
        "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
        "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
        "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
        "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
        "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
    ]

    COMMON_SURNAMES = [
        "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
        "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水",
        "山崎", "森", "池田", "橋本", "阿部", "石川", "山下", "中島", "石井", "小川",
        "前田", "岡田", "長谷川", "藤田", "後藤", "近藤", "村上", "遠藤", "青木", "坂本",
        "斉藤", "福田", "太田", "西村", "藤井", "金子", "岡本", "藤原", "中野", "三浦",
        "原田", "中田", "松田", "竹内", "中山", "和田", "石田", "上田", "森田", "原",
    ]

    PERSON_PREFIXES = ["お名前：", "お名前:", "氏名：", "氏名:", "担当者：", "担当者:", "Name:", "Dear "]
    ADDRESS_PREFIXES = ["ご住所：", "ご住所:", "住所：", "住所:", "所在地：", "所在地:", "Address:"]
    PERSON_SUFFIXES = ["様", "氏", "殿", "社長", "マネージャー"]

    def extract(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        entities.extend(self._extract_prefecture_addresses(text))
        entities.extend(self._extract_common_names(text))
        entities.extend(self._extract_prefixed(text))
        entities.extend(self._extract_suffix_names(text))
        entities.extend(self._extract_romanized_names(text))
        return entities

    def _extract_prefecture_addresses(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for pref in self.PREFECTURES:
            for match in re.finditer(re.escape(pref), text):
                start = match.start()
                post = text[match.end(): match.end() + 60]
                tail = re.match(r"[0-9０-９一-龠ぁ-んァ-ヶA-Za-z\-ー－−のがヶ\s,，.]{0,50}", post)
                suffix = tail.group(0) if tail else ""
                suffix = re.split(r"[。！？\n]", suffix, maxsplit=1)[0]
                suffix = re.sub(r"[はがにでもをと、，\s]+$", "", suffix)
                extracted = pref + suffix
                entities.append(
                    PIIEntity(PIIType.ADDRESS, extracted, start, start + len(extracted), 0.6, "dict_prefecture")
                )
        return entities

    def _extract_common_names(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for surname in self.COMMON_SURNAMES:
            for match in re.finditer(re.escape(surname), text):
                start = match.start()
                post = text[match.end(): match.end() + 7]
                given = re.match(r"[ 　]*[一-龠々ぁ-んァ-ヶ]{1,4}", post)
                extracted = surname
                score = 0.4
                if given:
                    candidate = given.group(0)
                    candidate = re.sub(r"[はがのにをともで、，。\s]+$", "", candidate)
                    if candidate.strip():
                        extracted += candidate
                        score = 0.5
                entities.append(
                    PIIEntity(PIIType.PERSON, extracted, start, start + len(extracted), score, "dict_name")
                )
        return entities

    def _extract_prefixed(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for prefix in self.PERSON_PREFIXES:
            for hit in re.finditer(re.escape(prefix), text):
                start = hit.end()
                window = text[start: start + 30].split("\n", 1)[0]
                match = re.match(r"[ 　]*[一-龠々ぁ-んァ-ヶA-Za-z.]+(?:[ 　]+[一-龠々ぁ-んァ-ヶA-Za-z.]+)?", window)
                if match and len(match.group(0).strip()) >= 2:
                    raw = match.group(0)
                    lead = len(raw) - len(raw.lstrip())
                    value = raw.strip()
                    ent_start = start + lead
                    entities.append(PIIEntity(PIIType.PERSON, value, ent_start, ent_start + len(value), 0.55, "dict_prefix"))

        for prefix in self.ADDRESS_PREFIXES:
            for hit in re.finditer(re.escape(prefix), text):
                start = hit.end()
                window = text[start: start + 80].split("\n", 1)[0]
                match = re.match(r"[ 　]*[0-9０-９一-龠々ぁ-んァ-ヶA-Za-z\-ー－−のがヶ\s,，.#]+", window)
                if match and len(match.group(0).strip()) >= 2:
                    raw = match.group(0)
                    lead = len(raw) - len(raw.lstrip())
                    value = re.sub(r"[。！？]+$", "", raw.strip())
                    ent_start = start + lead
                    entities.append(PIIEntity(PIIType.ADDRESS, value, ent_start, ent_start + len(value), 0.55, "dict_prefix"))
        return entities

    def _extract_suffix_names(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for suffix in self.PERSON_SUFFIXES:
            for hit in re.finditer(re.escape(suffix), text):
                before = text[max(0, hit.start() - 12): hit.start()]
                match = re.search(r"([一-龠々ぁ-んァ-ヶA-Za-z]+(?:[ 　]+[一-龠々ぁ-んァ-ヶA-Za-z]+)?)$", before)
                if match:
                    value = match.group(1)
                    start = hit.start() - len(value)
                    entities.append(PIIEntity(PIIType.PERSON, value, start, hit.start(), 0.5, "dict_suffix"))
        return entities

    def _extract_romanized_names(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        pattern = r"(?<![A-Za-z])[A-Z][a-z]+[ 　]+[A-Z][a-z]+(?![A-Za-z])"
        for match in re.finditer(pattern, text):
            entities.append(PIIEntity(PIIType.PERSON, match.group(0), match.start(), match.end(), 0.5, "dict_romanized"))
        return entities
