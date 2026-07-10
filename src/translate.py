"""
Multilingual flow (24h-scope version, per the brief: a translation step at
the boundary is acceptable).

query (any language) --detect--> if not English --translate--> English
                                                                    |
                                                            retrieval + LLM
                                                                    |
answer (original language) <--translate-- English answer <---------

We use langdetect for detection (fast, local, no API call) and the LLM
itself for translation (no extra library/API needed).
"""

from langdetect import detect, DetectorFactory
from src.llm import translate_text

DetectorFactory.seed = 0  # deterministic detection

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
}


def detect_language(text: str) -> str:
    try:
        code = detect(text)
    except Exception:
        code = "en"
    return code


def to_english(text: str, source_lang_code: str) -> str:
    if source_lang_code == "en":
        return text
    return translate_text(text, "English")


def from_english(text: str, target_lang_code: str) -> str:
    if target_lang_code == "en":
        return text
    target_name = LANGUAGE_NAMES.get(target_lang_code, target_lang_code)
    return translate_text(text, target_name)
