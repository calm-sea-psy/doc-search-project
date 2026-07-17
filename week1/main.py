import os
import sys

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tech_docs.csv")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")

    print("\n [#.1 데이터 불러오기]")
    print(f"데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
    
    return df


def explore_structure(df: pd.DataFrame) -> None:
    print("\n [#.2 데이터 구조 확인 ]")
    print(f"행 수: {df.shape[0]}, 열 수: {df.shape[1]}")

    print("\n=== 컬럼별 자료형 ===")
    print(df.dtypes)

    print("\n=== 상위 5행 ===")
    print(df.head(5))


def show_category_distribution(df: pd.DataFrame) -> dict:
    print("\n [#.3 카테고리 분포 확인 ]")

    total_count = len(df)
    counts = df["category"].value_counts()

    print("\n=== 카테고리별 문서 수 및 비율 ===")
    for cat, count in counts.items():
        ratio = count / total_count * 100
        print(f"{cat}: {count}개 ({ratio:.1f}%)")

    avg_word_counts = {}
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        word_counts = [len(text.split()) for text in cat_df["content"]]
        avg_word_counts[cat] = sum(word_counts) / len(word_counts)

    print("\n=== 카테고리별 평균 단어 수 ===")
    for cat, avg in avg_word_counts.items():
        print(f"{cat}: {avg:.1f}단어")

    return {
        "문서수": counts.to_dict(),
        "평균단어수": avg_word_counts,
    }


def check_missing(df: pd.DataFrame) -> dict:
    print("\n[#.4 결측치 현황 파악]")

    total_rows = len(df)
    missing_counts = df.isnull().sum()

    result = {}
    columns_with_missing = []
    columns_without_missing = []

    for col in df.columns:
        count = int(missing_counts[col])
        ratio = count / total_rows * 100

        if ratio < 5:
            severity = "낮음"
        elif ratio < 20:
            severity = "주의"
        else:
            severity = "높음"

        result[col] = {"결측치_수": count, "비율": ratio, "심각도": severity}

        if count > 0:
            columns_with_missing.append(col)
        else:
            columns_without_missing.append(col)

    print("\n=== 결측치가 있는 컬럼 ===")
    if columns_with_missing:
        for col in columns_with_missing:
            info = result[col]
            print(f"{col}: {info['결측치_수']}개 ({info['비율']:.1f}%) - 심각도: {info['심각도']}")
    else:
        print("결측치가 있는 컬럼: 없음")

    print("\n=== 결측치가 없는 컬럼 ===")
    print(", ".join(columns_without_missing))

    return result


def numpy_doc_stats(df: pd.DataFrame) -> None:
    print("\n[#.5 NumPy 문서 길이 통계량]")

    contents = df["content"].dropna()
    lengths = np.array([len(text.split()) for text in contents])

    mean_val = np.mean(lengths)
    std_val = np.std(lengths, ddof=1)
    median_val = np.median(lengths)
    min_val = np.min(lengths)
    max_val = np.max(lengths)

    print("\n=== NumPy 통계량 ===")
    print(f"평균: {mean_val:.2f}")
    print(f"표준편차: {std_val:.2f}")
    print(f"중앙값: {median_val:.2f}")
    print(f"최솟값: {min_val}")
    print(f"최댓값: {max_val}")

    short_docs = lengths[lengths < 50]
    print(f"\n=== 50단어 미만 문서 ===")
    print(f"개수: {len(short_docs)}개")
    if len(short_docs) > 0:
        print(f"단어 수 목록: {short_docs.tolist()}")

    print("\n=== pandas describe()와 비교 ===")
    pandas_stats = df["content"].apply(lambda t: len(t.split())).describe()
    #print(pandas_stats)
    print(f"NumPy 평균={mean_val:.2f} vs pandas 평균={pandas_stats['mean']:.2f}")
    print(f"NumPy 표준편차={std_val:.2f} vs pandas 표준편차={pandas_stats['std']:.2f}")
    print("\n")

def main() -> None:
    df = load_data(DATA_PATH)
    explore_structure(df)
    show_category_distribution(df)
    check_missing(df)
    numpy_doc_stats(df)


if __name__ == "__main__":
    main()
