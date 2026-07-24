import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ad_daily_stats.csv")
HEATMAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correlation_heatmap.png")

DATE_COLUMN = "날짜"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8")
    print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
    return df


def explore_data(df: pd.DataFrame) -> None:
    print("\n===== [1] 데이터 구조 확인 =====")
    print(f"행 수: {df.shape[0]}, 열 수: {df.shape[1]}")

    print("\n--- 상위 5행 ---")
    print(df.head())

    print("\n--- info() ---")
    df.info()

    print("\n--- describe() (기술통계) ---")
    print(df.describe())


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.drop(columns=[DATE_COLUMN])
    corr_matrix = numeric_df.corr(method="pearson")

    print("\n===== [2] 상관행렬 (Pearson) =====")
    print(corr_matrix.round(3))

    return corr_matrix


def plot_heatmap(corr_matrix: pd.DataFrame, save_path: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("광고 성과 지표 간 상관관계 히트맵")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"\n===== [3] 히트맵 저장 완료: {save_path} =====")


def top_correlated(corr_matrix: pd.DataFrame, target: str, top_n: int = 3) -> pd.Series:
    correlations = corr_matrix[target].drop(target)
    ranked_index = correlations.abs().sort_values(ascending=False).index
    return correlations.reindex(ranked_index).head(top_n)


def main() -> None:
    df = load_data(DATA_PATH)
    explore_data(df)

    corr_matrix = compute_correlation(df)
    plot_heatmap(corr_matrix, HEATMAP_PATH)

    print("\n===== [4] 수익률 기준 상관관계 Top 3 =====")
    for name, value in top_correlated(corr_matrix, "수익률").items():
        print(f"{name}: {value:.3f}")

    print("\n===== [5] 전환율 기준 상관관계 Top 3 =====")
    for name, value in top_correlated(corr_matrix, "전환율").items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
