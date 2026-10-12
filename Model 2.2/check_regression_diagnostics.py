import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load training data (required to calculate VIF)
train_df = pd.read_csv("model2_2_train.csv")

feature_cols = [
    "fifa_points_diff",
    "fifa_trend_1y_diff",
    "host_diff",
    "rest_days_diff",
    "team_xg90_pre",
    "team_sot90_pre",
    "opp_xga90_pre",
    "opp_save_pct_pre",
]

X_train = train_df[feature_cols]

# 2. VIF Calculation
vif_data = pd.DataFrame()
vif_data["Feature"] = feature_cols
vif_data["VIF"] = [
    variance_inflation_factor(X_train.values, i)
    for i in range(len(feature_cols))
]
vif_data.to_csv("vif_results.csv", index=False)

print("VIF results calculated and saved to 'vif_results.csv'.")