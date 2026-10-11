"""
HIT140 Objective 2 - Linear Regression 2.1
Predict the GOAL DIFFERENCE (Team A goals - Team B goals) of FIFA World Cup 2026 matches.

Input : model1_data.csv  (104 rows = 104 matches, 8 explanatory variables, target goal_diff)
Output: outputs_2.1/     (results.txt, CSV tables and PNG charts for the report)
Run   : python model_2_1.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X_COLS = ["fifa_points_diff", "fifa_trend_1y_diff", "host_diff", "rest_days_diff",
          "strong_conf_diff", "wc2022_stage_diff", "prior_gf_pg_diff", "prior_ga_pg_diff"]
Y_COL = "goal_diff"
OUT = "outputs_2.1"
os.makedirs(OUT, exist_ok=True)
log_file = open(f"{OUT}/results.txt", "w", encoding="utf-8")

def log(*a):
    print(*a)
    print(*a, file=log_file)

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=150)
    plt.close(fig)

# =====================================================================
# 1. LOAD DATA AND CHECK REQUIREMENTS
# =====================================================================
df = pd.read_csv("model1_data.csv")
X, y = df[X_COLS], df[Y_COL]
log("1. DATA CHECKS")
checks = {
    "104 rows (one per match)": len(df) == 104,
    "match_id unique": df["match_id"].is_unique,
    "exactly 8 explanatory variables": len(X_COLS) == 8,
    "no missing values": df[X_COLS + [Y_COL]].isna().sum().sum() == 0,
    "matchday-1 form differences = 0 (no future information)": (df.loc[df.match_id <= 24, ["prior_gf_pg_diff", "prior_ga_pg_diff"]] == 0).all().all(),
}
for k, v in checks.items():
    log(f"   [{'OK' if v else 'PROBLEM'}] {k}")

# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =====================================================================
log("\n2. EXPLORATORY DATA ANALYSIS")
desc = df[X_COLS + [Y_COL]].describe().T[["mean", "std", "min", "max"]].round(3)
desc.to_csv(f"{OUT}/table1_descriptive_stats.csv")
log(desc.to_string())
log(f"\nTeam A won {(y > 0).sum()}, drew {(y == 0).sum()}, lost {(y < 0).sum()} matches")

corr = df[X_COLS + [Y_COL]].corr().round(3)
corr.to_csv(f"{OUT}/table2_correlation_matrix.csv")
log("\nCorrelation with goal_diff:")
log(corr[Y_COL].drop(Y_COL).sort_values(ascending=False).to_string())

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im); ax.set_title("Correlation matrix (Model 2.1)")
save_fig(fig, "fig1_correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(y, bins=range(y.min(), y.max() + 2), edgecolor="black", align="left")
ax.set_xlabel("Goal difference (Team A - Team B)"); ax.set_ylabel("Number of matches")
ax.set_title("Distribution of goal difference")
save_fig(fig, "fig2_goal_diff_histogram.png")

fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
for ax, col in zip(axes.ravel(), X_COLS):
    ax.scatter(df[col], y, alpha=0.6, s=18)
    ax.set_xlabel(col); ax.set_ylabel("goal_diff"); ax.set_title(f"r = {corr.loc[col, Y_COL]:.2f}", fontsize=10)
fig.suptitle("Each explanatory variable vs goal difference")
save_fig(fig, "fig3_scatter_each_variable.png")

# =====================================================================
# 3. TRAIN/TEST SPLIT AND OLS MODEL
# =====================================================================
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"\n3. OLS MODEL  (train = {len(X_tr)} matches, test = {len(X_te)} matches, random_state = 42)")
df.loc[X_tr.index].to_csv(f"{OUT}/train_2_1.csv", index=False)
df.loc[X_te.index].to_csv(f"{OUT}/test_2_1.csv", index=False)
model = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
log(model.summary().as_text())
with open(f"{OUT}/ols_model_summary.txt", "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

coef = pd.DataFrame({"coefficient": model.params, "std_error": model.bse, "t": model.tvalues,
                     "p_value": model.pvalues, "ci_2.5%": model.conf_int()[0],
                     "ci_97.5%": model.conf_int()[1]}).round(4)
coef.to_csv(f"{OUT}/table3_coefficients.csv")
sig = [c for c in X_COLS if coef.loc[c, "p_value"] < 0.05]
log(f"\nSignificant at 5%: {', '.join(sig) if sig else 'none'}")

# =====================================================================
# 4. ASSUMPTION CHECKS
# =====================================================================
log("\n4. ASSUMPTION CHECKS")
Xc = sm.add_constant(X_tr)
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])], index=X_COLS).round(2)
vif.to_csv(f"{OUT}/table4_vif.csv", header=["VIF"])
log("Multicollinearity (VIF > 5 moderate, > 10 serious):"); log(vif.to_string())
bp = het_breuschpagan(model.resid, model.model.exog)[1]
sw = stats.shapiro(model.resid).pvalue
dw = sm.stats.durbin_watson(model.resid)
log(f"Homoscedasticity  Breusch-Pagan p = {bp:.4f}  -> {'OK' if bp > 0.05 else 'heteroscedasticity detected'}")
log(f"Normality         Shapiro-Wilk  p = {sw:.4f}  -> {'OK' if sw > 0.05 else 'residuals not normal'}")
log(f"Independence      Durbin-Watson  = {dw:.3f}   -> {'OK' if 1.5 < dw < 2.5 else 'check'}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(model.fittedvalues, model.resid, alpha=0.7); ax.axhline(0, color="red", ls="--")
ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs fitted")
save_fig(fig, "fig4_residuals_vs_fitted.png")
fig = sm.qqplot(model.resid, line="45", fit=True); plt.title("Normal Q-Q plot of residuals")
save_fig(fig, "fig5_qq_plot.png")

# =====================================================================
# 5. EVALUATION (test set, baseline, cross-validation, reduced model)
# =====================================================================
def evaluate(m, cols):
    p_tr = m.predict(sm.add_constant(X_tr[cols]))
    p_te = m.predict(sm.add_constant(X_te[cols], has_constant="add"))
    cv = [r2_score(y.iloc[te], LinearRegression().fit(X.iloc[tr][cols], y.iloc[tr]).predict(X.iloc[te][cols]))
          for tr, te in KFold(5, shuffle=True, random_state=42).split(X)]
    return {"R2 train": r2_score(y_tr, p_tr), "Adj R2 train": m.rsquared_adj, "R2 test": r2_score(y_te, p_te),
            "RMSE test": np.sqrt(mean_squared_error(y_te, p_te)), "MAE test": mean_absolute_error(y_te, p_te),
            "CV R2 (5-fold mean)": np.mean(cv)}, p_te

full, p_te = evaluate(model, X_COLS)
base = np.full(len(y_te), y_tr.mean())
baseline = {"R2 test": r2_score(y_te, base), "RMSE test": np.sqrt(mean_squared_error(y_te, base)),
            "MAE test": mean_absolute_error(y_te, base)}
columns = {"Full model (8 variables)": full}
if vif.max() > 5:                                     # only if multicollinearity is a problem
    drop = vif.idxmax()
    reduced_cols = [c for c in X_COLS if c != drop]
    reduced, _ = evaluate(sm.OLS(y_tr, sm.add_constant(X_tr[reduced_cols])).fit(), reduced_cols)
    columns[f"Without {drop}"] = reduced
columns["Baseline (predict mean)"] = baseline
ev = pd.DataFrame(columns).round(3)
ev.to_csv(f"{OUT}/table5_evaluation.csv")
pred = df.loc[X_te.index, ["match_id", "team_a", "team_b"]].copy()
pred["actual_goal_diff"] = y_te
pred["predicted_goal_diff"] = p_te.round(3)
pred["residual"] = (y_te - p_te).round(3)
pred.to_csv(f"{OUT}/test_predictions.csv", index=False)
log("\n5. MODEL EVALUATION"); log(ev.to_string())

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_te, p_te, alpha=0.8); ax.plot([-5, 7], [-5, 7], "r--")
ax.set_xlim(-5, 7); ax.set_ylim(-5, 7)
ax.set_xlabel("Actual goal difference"); ax.set_ylabel("Predicted goal difference")
ax.set_title("Actual vs predicted (test set)")
save_fig(fig, "fig6_actual_vs_predicted.png")

log(f"\nAll tables and charts saved in folder: {OUT}/")
log_file.close()
