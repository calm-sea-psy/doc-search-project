import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tech_docs.csv")

EVAL_SET = [
    {"query": "how to create a list from another list in python", "relevant_doc_ids": ["D001"]},
    {"query": "what is a python decorator", "relevant_doc_ids": ["D010"]},
    {"query": "how to handle exceptions in python", "relevant_doc_ids": ["D005"]},
    {"query": "how to resolve merge conflicts in git", "relevant_doc_ids": ["D018"]},
    {"query": "how to keep a linear commit history in git", "relevant_doc_ids": ["D015"]},
    {"query": "how to temporarily save uncommitted changes in git", "relevant_doc_ids": ["D019"]},
    {"query": "how does gradient descent work in machine learning", "relevant_doc_ids": ["D023"]},
    {"query": "what causes overfitting in machine learning models", "relevant_doc_ids": ["D026"]},
    {"query": "how does backpropagation work in neural networks", "relevant_doc_ids": ["D030"]},
    {"query": "what are common regularization techniques in neural networks", "relevant_doc_ids": ["D027", "D056"]},
    {"query": "different ways to select or filter elements in a numpy array", "relevant_doc_ids": ["D033", "D039", "D057"]},
    {"query": "what is broadcasting in numpy", "relevant_doc_ids": ["D034"]},
    {"query": "how to handle missing data in pandas", "relevant_doc_ids": ["D043"]},
    {"query": "how to split data into groups and calculate aggregation statistics in pandas", "relevant_doc_ids": ["D044"]},
    {"query": "how to merge two dataframes in pandas", "relevant_doc_ids": ["D045"]},
]


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
    return df


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def keyword_search(question: str, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    question_words = set(preprocess(question).split())

    scores = []
    for content in df["content_clean"]:
        doc_words = set(content.split())
        scores.append(len(question_words & doc_words))

    result = df[["doc_id", "title", "category"]].copy()
    result["score"] = scores
    result = result.sort_values("score", ascending=False)
    return result.head(top_k)


def build_tfidf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["content_clean"]).toarray()

    n_docs, n_words = tfidf_matrix.shape
    print(f"TF-IDF 행렬 크기: ({n_docs}, {n_words}) | 사용된 단어 수: {n_words}")

    return tfidf_matrix, vectorizer


def tfidf_search(question: str, df: pd.DataFrame, tfidf_matrix, vectorizer, top_k: int = 3) -> pd.DataFrame:
    question_vec = vectorizer.transform([preprocess(question)]).toarray()[0]

    similarities = np.array([
        cosine_similarity_numpy(question_vec, doc_vec) for doc_vec in tfidf_matrix
    ])

    top_indices = similarities.argsort()[::-1][:top_k]

    result = df.iloc[top_indices][["doc_id", "title", "category"]].copy()
    result["similarity"] = similarities[top_indices].round(4)
    return result


def precision_at_k(result_ids: list, relevant_ids: list, k: int) -> float:
    top_k_ids = set(result_ids[:k])
    hits = top_k_ids & set(relevant_ids)
    return len(hits) / k


def reciprocal_rank(result_ids: list, relevant_ids: list) -> float:
    for rank, doc_id in enumerate(result_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0.0


def run_evaluation(eval_set: list, search_fn, k: int) -> dict:
    precisions = []
    reciprocal_ranks = []

    for item in eval_set:
        result_ids = search_fn(item["query"])["doc_id"].tolist()
        precisions.append(precision_at_k(result_ids, item["relevant_doc_ids"], k))
        reciprocal_ranks.append(reciprocal_rank(result_ids, item["relevant_doc_ids"]))

    return {
        f"Precision@{k}": np.mean(precisions),
        "MRR": np.mean(reciprocal_ranks),
    }


def analyze_failures(eval_set: list, search_fn, k: int) -> None:
    for item in eval_set:
        result_ids = search_fn(item["query"])["doc_id"].tolist()

        if reciprocal_rank(result_ids, item["relevant_doc_ids"]) == 0.0:
            print(f"질문: {item['query']}")
            print(f"  정답 doc_id : {item['relevant_doc_ids']}")
            print(f"  검색 결과   : {result_ids[:k]}")


def main() -> None:
    df = load_data(DATA_PATH)
    df["content_clean"] = df["content"].dropna().apply(preprocess)
    tfidf_matrix, vectorizer = build_tfidf(df)

    print(f"\n평가셋 크기: {len(EVAL_SET)}개 질문")

    k = 3
    baseline_fn = lambda q: keyword_search(q, df, top_k=k)
    tfidf_fn = lambda q: tfidf_search(q, df, tfidf_matrix, vectorizer, top_k=k)

    baseline_metrics = run_evaluation(EVAL_SET, baseline_fn, k)
    tfidf_metrics = run_evaluation(EVAL_SET, tfidf_fn, k)

    comparison = pd.DataFrame(
        [baseline_metrics, tfidf_metrics],
        index=["Keyword Baseline", "TF-IDF"],
    )
    print("\n=== 성능 비교 ===")
    print(comparison.round(4).to_string())

    print("\n=== 실패 케이스 (Top-3 안에 정답 없음) ===")
    analyze_failures(EVAL_SET, tfidf_fn, k)


if __name__ == "__main__":
    main()
