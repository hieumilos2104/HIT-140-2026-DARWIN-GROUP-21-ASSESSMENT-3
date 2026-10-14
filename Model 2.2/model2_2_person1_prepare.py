

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "model2_2_data.csv"
RESULTS_DIR = PROJECT_DIR / "model2_2_results"

FEATURES = [
    "fifa_points_diff",
    "fifa_trend_1y_diff",
    "host_diff",
    "rest_days_diff",
    "team_xg90_pre",
    "team_sot90_pre",
    "opp_xga90_pre",
    "opp_save_pct_pre",
]
TARGET = "goals_scored"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    required_columns = ["match_id", "date", "stage", "team", "opponent", *FEATURES, TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Assignment requirement checks.
    if len(df) != 208:
        raise ValueError(f"Expected 208 rows, found {len(df)}")
    if df["match_id"].nunique() != 104:
        raise ValueError(f"Expected 104 matches, found {df['match_id'].nunique()}")
    if not df.groupby("match_id").size().eq(2).all():
        raise ValueError("Every match_id must have exactly two team rows")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("The explanatory variables or target contain missing values")

    print("Dataset checks passed")
    print(f"Rows: {len(df)}")
    print(f"Matches: {df['match_id'].nunique()}")
    print(f"Explanatory variables: {len(FEATURES)}")
    print("\nDescriptive statistics:")
    print(df[FEATURES + [TARGET]].describe().round(3))

    # Save descriptive statistics for the report.
    df[FEATURES + [TARGET]].describe().round(3).to_csv(
        RESULTS_DIR / "descriptive_statistics.csv"
    )

    # Correlation matrix and heatmap.
    correlations = df[FEATURES + [TARGET]].corr()
    correlations.round(3).to_csv(RESULTS_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlations.index)), correlations.index)
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title("Correlation Matrix: Linear Regression 2.2")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # Keep both teams from the same match in the same partition.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_indices, test_indices = next(
        splitter.split(df, y=df[TARGET], groups=df["match_id"])
    )
    train_df = df.iloc[train_indices].sort_values(["match_id", "team"]).reset_index(drop=True)
    test_df = df.iloc[test_indices].sort_values(["match_id", "team"]).reset_index(drop=True)

    train_matches = set(train_df["match_id"])
    test_matches = set(test_df["match_id"])
    if train_matches & test_matches:
        raise ValueError("A match appears in both training and testing data")

    train_df.to_csv(RESULTS_DIR / "model2_2_train.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "model2_2_test.csv", index=False)

    print("\nGroup-based train/test split saved")
    print(f"Training: {len(train_df)} rows from {len(train_matches)} matches")
    print(f"Testing: {len(test_df)} rows from {len(test_matches)} matches")
    print(f"Output folder: {RESULTS_DIR}")



