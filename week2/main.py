import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tech_docs.csv")
DEMO_QUESTION = "how does gradient descent work in machine learning"


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


def tfidf_search(question: str, df: pd.DataFrame, tfidf_matrix, vectorizer, top_k: int = 3) -> pd.DataFrame:
    question_vec = vectorizer.transform([preprocess(question)]).toarray()[0]

    similarities = np.array([
        cosine_similarity_numpy(question_vec, doc_vec) for doc_vec in tfidf_matrix
    ])

    top_indices = similarities.argsort()[::-1][:top_k]

    result = df.iloc[top_indices][["doc_id", "title", "category"]].copy()
    result["similarity"] = similarities[top_indices].round(4)
    return result


def build_tfidf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["content_clean"]).toarray()

    n_docs, n_words = tfidf_matrix.shape
    print(f"TF-IDF 행렬 크기: ({n_docs}, {n_words}) | 사용된 단어 수: {n_words}")

    return tfidf_matrix, vectorizer

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

# Baseline과 TF-IDF 결과 비교 
def main() -> None:
    df = load_data(DATA_PATH)

    df["content_clean"] = df["content"].dropna().apply(preprocess)
    print("전처리 완료: content_clean 컬럼 생성")

    tfidf_matrix, vectorizer = build_tfidf(df)

    print(f"\n질문: {DEMO_QUESTION}")

    print("\n=== Keyword Baseline ===")
    print(keyword_search(DEMO_QUESTION, df).to_string(index=False))

    print("\n=== TF-IDF Search ===")
    print(tfidf_search(DEMO_QUESTION, df, tfidf_matrix, vectorizer).to_string(index=False))

    
if __name__ == "__main__":
    main()
