"""
Гибридный RAG: TF-IDF + эмбеддинги (all-MiniLM-L6-v2) + RRF.
Contextual Retrieval: чанки дополняются кратким контекстом перед индексацией.
Опора: глава 6 книги «AI-инженерия» (Чип Хьюен).
"""

import os
import re
import math
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === 1. Загрузка и чанкинг с контекстом ===
def load_and_chunk(docs_dir="data/docs"):
    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            # Contextual Retrieval: добавляем краткий контекст
            context = f"Это фрагмент из документа {filename} (абзац {i+1})."
            chunks.append({
                "id": f"{filename}::chunk_{i}",
                "source": filename,
                "text": p,
                "context": context,
                "indexed_text": f"{context} {p}",
            })
    return chunks

# === 2. TF-IDF поиск ===
def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_tfidf_index(chunks):
    doc_freq = Counter()
    for chunk in chunks:
        for t in set(tokenize(chunk["indexed_text"])):
            doc_freq[t] += 1
    return doc_freq

def tfidf_scores(query, chunks, doc_freq):
    query_tokens = tokenize(query)
    total = len(chunks)
    scores = []
    for chunk in chunks:
        tokens = tokenize(chunk["indexed_text"])
        tf = Counter(tokens)
        score = 0.0
        for qt in query_tokens:
            if qt in tf:
                idf = math.log((total + 1) / (doc_freq.get(qt, 0) + 1)) + 1
                score += tf[qt] * idf
        scores.append(score)
    return np.array(scores)

# === 3. Эмбеддинг-поиск ===
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts):
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def embedding_scores(query, chunks, chunk_embeddings):
    q_emb = embed_texts([query])
    sims = cosine_similarity(q_emb, chunk_embeddings)[0]
    return sims

# === 4. RRF ===
def rrf_fuse(scores_list, k=60):
    """Reciprocal Rank Fusion: объединяет несколько ранжирований."""
    n = len(scores_list[0])
    fused = np.zeros(n)
    for scores in scores_list:
        ranks = np.argsort(-scores)  # индексы от лучшего к худшему
        for rank_pos, idx in enumerate(ranks):
            fused[idx] += 1.0 / (k + rank_pos + 1)
    return fused

# === 5. Гибридный поиск (нормализация + взвешенная сумма) ===
def normalize(scores):
    """Min-max нормализация в [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)

def hybrid_search(query, chunks, doc_freq, chunk_embeddings, top_k=3,
                  w_tfidf=0.3, w_emb=0.7):
    tf_scores = normalize(tfidf_scores(query, chunks, doc_freq))
    emb_scores = normalize(embedding_scores(query, chunks, chunk_embeddings))
    combined = w_tfidf * tf_scores + w_emb * emb_scores
    order = np.argsort(-combined)[:top_k]
    return [(combined[i], chunks[i]) for i in order]

# === 6. Генерация (демо) ===
def generate_answer(query, top_chunks):
    if not top_chunks:
        return "Не найдено релевантных документов."
    answer = f"Вопрос: {query}\n\nНайденные фрагменты (гибридный поиск):\n"
    for i, (score, chunk) in enumerate(top_chunks, 1):
        answer += f"\n[{i}] Источник: {chunk['source']} (RRF-оценка: {score:.4f})\n"
        answer += f"    {chunk['text'][:200]}...\n"
    answer += "\n(Демо-режим. В реальности здесь был бы ответ LLM.)"
    return answer

if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"Загружено чанков: {len(chunks)}")
    doc_freq = build_tfidf_index(chunks)
    print("Строим эмбеддинги...")
    chunk_embeddings = embed_texts([c["indexed_text"] for c in chunks])
    print(f"Эмбеддинги: {chunk_embeddings.shape}")

    queries = [
        "Как создать функцию в Python?",
        "Как создать ветку в Git?",
        "Как собрать Docker образ?",
        "Как обработать исключение?",
        "Как отправить изменения в удалённый репозиторий?",
    ]

    os.makedirs("results", exist_ok=True)
    with open("results/answers_hybrid.txt", "w", encoding="utf-8") as f:
        for q in queries:
            top = hybrid_search(q, chunks, doc_freq, chunk_embeddings, top_k=3)
            answer = generate_answer(q, top)
            print("=" * 60)
            print(answer)
            f.write("=" * 60 + "\n" + answer + "\n\n")
    print("\nРезультаты сохранены в results/answers_hybrid.txt")
