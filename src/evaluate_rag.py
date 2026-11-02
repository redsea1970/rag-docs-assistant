"""
Оценка RAG: context_precision и faithfulness (упрощённые, без API).
Опора: глава 4 книги «AI-инженерия» (Чип Хьюен).
"""

import os
import re
import math
from collections import Counter

import numpy as np

from hybrid_rag import (
    load_and_chunk, build_tfidf_index, embed_texts, hybrid_search,
)


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def overlap_ratio(a_tokens, b_tokens):
    """Доля общих токенов (пересечение / |a|)."""
    if not a_tokens:
        return 0.0
    common = set(a_tokens) & set(b_tokens)
    return len(common) / len(a_tokens)


def context_precision(query, chunks):
    """Доля найденных чанков, релевантных запросу (по пересечению токенов)."""
    q_tokens = tokenize(query)
    relevant = 0
    for _, chunk in chunks:
        if overlap_ratio(q_tokens, tokenize(chunk["text"])) > 0:
            relevant += 1
    return relevant / len(chunks) if chunks else 0.0


def faithfulness(query, chunks):
    """Насколько найденные чанки покрывают ключевые термины запроса."""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    context_tokens = set()
    for _, chunk in chunks:
        context_tokens |= set(tokenize(chunk["text"]))
    return len(q_tokens & context_tokens) / len(q_tokens)


def main():
    queries = [
        "Как создать функцию в Python?",
        "Как создать ветку в Git?",
        "Как собрать Docker образ?",
        "Как обработать исключение?",
        "Как отправить изменения в удалённый репозиторий?",
    ]

    chunks = load_and_chunk()
    doc_freq = build_tfidf_index(chunks)
    chunk_embeddings = embed_texts([c["indexed_text"] for c in chunks])

    print(f"{'Запрос':<50} {'Precision':>10} {'Faithfulness':>14}")
    print("-" * 78)
    precisions, faithfulnesses = [], []
    for q in queries:
        top = hybrid_search(q, chunks, doc_freq, chunk_embeddings, top_k=3)
        p = context_precision(q, top)
        f = faithfulness(q, top)
        precisions.append(p)
        faithfulnesses.append(f)
        print(f"{q[:48]:<50} {p:>10.2f} {f:>14.2f}")

    print("-" * 78)
    print(f"{'Среднее':<50} {np.mean(precisions):>10.2f} {np.mean(faithfulnesses):>14.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.txt", "w", encoding="utf-8") as f:
        f.write("Оценка RAG (context_precision, faithfulness)\n")
        f.write("=" * 60 + "\n")
        for q, p, fa in zip(queries, precisions, faithfulnesses):
            f.write(f"{q}\n  Precision: {p:.2f}, Faithfulness: {fa:.2f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Среднее: Precision={np.mean(precisions):.2f}, Faithfulness={np.mean(faithfulnesses):.2f}\n")
    print("\nРезультаты сохранены в results/evaluation.txt")


if __name__ == "__main__":
    main()
