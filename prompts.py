# prompts.py
# Все промпты проекта вынесены сюда для версионирования и переиспользования.
# Формат: словарь с метаданными (модель, дата создания, автор, версия).

from datetime import date

PROMPTS = {
    "baseline_qa": {
        "text": "Ответь на вопрос: {question}",
        "model": "demo",
        "date_created": str(date.today()),
        "author": "redsea1970",
        "version": "1.0",
        "description": "Базовый промпт без инструкций и примеров."
    },
    "ricce_qa": {
        "text": (
            "Роль: ты — эксперт по AI-инженерии.\n"
            "Инструкция: отвечай кратко, по делу, с примерами.\n"
            "Контекст: вопрос пользователя ниже.\n"
            "Ограничения: не выдумывай факты, если не уверен — скажи об этом.\n"
            "Вопрос: {question}"
        ),
        "model": "demo",
        "date_created": str(date.today()),
        "author": "redsea1970",
        "version": "1.0",
        "description": "RICCE-промпт: роль, инструкция, контекст, ограничения, вопрос."
    },
}

def get_prompt(name: str, **kwargs) -> str:
    """Возвращает текст промпта по имени, подставляя параметры."""
    if name not in PROMPTS:
        raise KeyError(f"Промпт '{name}' не найден. Доступны: {list(PROMPTS)}")
    return PROMPTS[name]["text"].format(**kwargs)
