# guardrails.py
# Простые ограничивающие механизмы (guardrails) для AI-приложений.
# Опора: глава 10 книги «AI-инженерия» (Чип Хьюен).

import re
import json

# --- Фильтр PII (персональные данные) ---
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")

# --- Простой список токсичных слов (демо, без API) ---
TOXIC_WORDS = {"дурак", "идиот", "ненавижу", "убей", "бомба"}

def contains_pii(text: str) -> bool:
    """Проверяет, есть ли в тексте email или телефон."""
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text))

def mask_pii(text: str) -> str:
    """Заменяет PII на метки-заполнители."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text

def is_toxic(text: str) -> bool:
    """Проверяет, есть ли в тексте токсичные слова."""
    lowered = text.lower()
    return any(word in lowered for word in TOXIC_WORDS)

def is_valid_json(text: str) -> bool:
    """Проверяет, является ли текст корректным JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def apply_guardrails(text: str) -> dict:
    """Прогоняет текст через все фильтры и возвращает отчёт."""
    return {
        "has_pii": contains_pii(text),
        "masked_text": mask_pii(text),
        "is_toxic": is_toxic(text),
        "is_valid_json": is_valid_json(text),
    }
