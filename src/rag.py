import os
import re
import math
from collections import Counter

# === 1. Загрузка и чанкинг документов ===
def load_and_chunk(docs_dir="data/docs", chunk_size=200):
    chunks = []
    for filename in os.listdir(docs_dir):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        # Разбиваем на чанки по абзацам
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            chunks.append({
                "id": f"{filename}::chunk_{i}",
                "source": filename,
                "text": p
            })
    return chunks

# === 2. TF-IDF поиск ===
def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_index(chunks):
    doc_freq = Counter()
    for chunk in chunks:
        tokens = set(tokenize(chunk["text"]))
        for t in tokens:
            doc_freq[t] += 1
    return doc_freq

def score_chunk(query_tokens, chunk, doc_freq, total_docs):
    tokens = tokenize(chunk["text"])
    tf = Counter(tokens)
    score = 0.0
    for qt in query_tokens:
        if qt in tf:
            idf = math.log((total_docs + 1) / (doc_freq.get(qt, 0) + 1)) + 1
            score += tf[qt] * idf
    return score

def search(query, chunks, doc_freq, top_k=3):
    query_tokens = tokenize(query)
    scored = [(score_chunk(query_tokens, c, doc_freq, len(chunks)), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# === 3. Генерация ответа (демо-режим) ===
def generate_answer(query, top_chunks):
    if not top_chunks or top_chunks[0][0] == 0:
        return "Не найдено релевантных документов."
    answer = f"Вопрос: {query}\n\n"
    answer += "Найденные фрагменты:\n"
    for i, (score, chunk) in enumerate(top_chunks, 1):
        answer += f"\n[{i}] Источник: {chunk['source']} (релевантность: {score:.2f})\n"
        answer += f"    {chunk['text'][:200]}...\n"
    answer += "\n(Это демо-режим. В реальности здесь был бы сгенерированный ответ от LLM на основе найденных фрагментов.)"
    return answer

# === 4. Основной запуск ===
if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"Загружено чанков: {len(chunks)}")
    doc_freq = build_index(chunks)

    queries = [
        "Как создать функцию в Python?",
        "Как создать ветку в Git?",
        "Как собрать Docker образ?",
        "Как обработать исключение?",
        "Как отправить изменения в удалённый репозиторий?",
    ]

    os.makedirs("results", exist_ok=True)
    with open("results/answers.txt", "w", encoding="utf-8") as f:
        for q in queries:
            top = search(q, chunks, doc_freq, top_k=3)
            answer = generate_answer(q, top)
            print("=" * 60)
            print(answer)
            f.write("=" * 60 + "\n" + answer + "\n\n")
    print("\nРезультаты сохранены в results/answers.txt")
