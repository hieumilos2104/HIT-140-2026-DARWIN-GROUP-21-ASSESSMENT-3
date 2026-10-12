import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data & Train OLS Model
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
target_col = "goals_scored"

X_train = train_df[feature_cols]
y_train = train_df[target_col]

# Add constant for OLS intercept
X_train_sm = sm.add_constant(X_train)

# Fit OLS Model & Save Summary
model = sm.OLS(y_train, X_train_sm).fit()
with open("ols_model_summary.txt", "w") as f:
  f.write(model.summary().as_text())


# Test Set Predictions & Evaluation
test_df = pd.read_csv("model2_2_test.csv")

X_test = test_df[feature_cols]
y_test = test_df[target_col]
X_test_sm = sm.add_constant(X_test)

# Generate test predictions
y_pred_test = model.predict(X_test_sm)

# Calculate evaluation metrics
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Save metrics and predictions to CSV
eval_df = pd.DataFrame(
    {"Metric": ["R-squared", "MAE", "RMSE"], "Value": [test_r2, test_mae, test_rmse]}
)
eval_df.to_csv("evaluation_metrics.csv", index=False)

preds_df = test_df[["match_id", "team", "opponent", "goals_scored"]].copy()
preds_df["predicted_goals"] = y_pred_test
preds_df.to_csv("test_predictions.csv", index=False)