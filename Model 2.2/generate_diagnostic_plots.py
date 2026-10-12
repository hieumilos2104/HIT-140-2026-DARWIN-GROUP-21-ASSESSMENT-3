import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Load training data & fit model to get residuals
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

X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()
train_residuals = model.resid

# Residuals vs Fitted Plot
plt.figure(figsize=(8, 6))
plt.scatter(model.fittedvalues, train_residuals, alpha=0.7, edgecolor="k")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Fitted Values (Predicted Goals)")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig("residuals_vs_fitted.png", dpi=300, bbox_inches="tight")
plt.close()

# Residual Normal Q-Q Plot
plt.figure(figsize=(8, 6))
stats.probplot(train_residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot of Residuals")
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig("residual_qq_plot.png", dpi=300, bbox_inches="tight")
plt.close()