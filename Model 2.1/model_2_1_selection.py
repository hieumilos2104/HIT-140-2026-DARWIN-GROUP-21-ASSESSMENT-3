"""
HIT140 Objective 2 - Linear Regression 2.1 (supplement)
Variable-selection evidence and a compact figure for the report.

Run AFTER model_2_1.py (uses the same data, split and random_state).
Input : model1_data.csv
Output: outputs_2.1/table6_selection_steps.txt
        outputs_2.1/table7_full_vs_fifa_only.csv
        outputs_2.1/table8_report_coefficients.csv
        outputs_2.1/fig7_diagnostics_combined.png
Run   : python model_2_1_selection.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X_COLS = ["fifa_points_diff", "fifa_trend_1y_diff", "host_diff", "rest_days_diff",
          "strong_conf_diff", "wc2022_stage_diff", "prior_gf_pg_diff", "prior_ga_pg_diff"]
Y_COL = "goal_diff"
OUT = "outputs_2.1"
ALPHA = 0.05
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("model1_data.csv")
X, y = df[X_COLS], df[Y_COL]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)   # same split as model_2_1.py


def ols(cols):
    return sm.OLS(y_tr, sm.add_constant(X_tr[cols])).fit()


lines = []

# ---------- 1. Backward elimination (drop the largest p-value until all p < 0.05) ----------
lines.append("BACKWARD ELIMINATION (training set, alpha = 0.05)")
cols = X_COLS.copy()
while True:
    p = ols(cols).pvalues.drop("const")
    if p.max() < ALPHA:
        break
    lines.append(f"  removed {p.idxmax():20s} p = {p.max():.3f}  -> {len(cols) - 1} variables left")
    cols.remove(p.idxmax())
lines.append(f"  final model: {cols}")
backward_final = cols

# ---------- 2. Forward selection (add the smallest p-value while p < 0.05) ----------
lines.append("\nFORWARD SELECTION (training set, alpha = 0.05)")
selected, remaining = [], X_COLS.copy()
while remaining:
    p = {c: ols(selected + [c]).pvalues[c] for c in remaining}
    best = min(p, key=p.get)
    if p[best] >= ALPHA:
        lines.append(f"  stop: best remaining candidate {best} has p = {p[best]:.3f}")
        break
    lines.append(f"  added {best:20s} p = {p[best]:.4f}")
    selected.append(best)
    remaining.remove(best)
lines.append(f"  final model: {selected}")

# ---------- 3. Joint test: do the other 7 variables add anything together? ----------
full, reduced = ols(X_COLS), ols(["fifa_points_diff"])
f_stat, f_p, df_diff = full.compare_f_test(reduced)
lines.append(f"\nNESTED F-TEST (full 8 vs fifa_points_diff only): "
             f"F({int(df_diff)}, {int(full.df_resid)}) = {f_stat:.2f}, p = {f_p:.3f}")

with open(f"{OUT}/table6_selection_steps.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))


# ---------- 4. Compare the two models on every criterion ----------
def evaluate(cols):
    m = ols(cols)
    p_te = m.predict(sm.add_constant(X_te[cols], has_constant="add"))
    cv5 = [r2_score(y.iloc[te], LinearRegression().fit(X.iloc[tr][cols], y.iloc[tr]).predict(X.iloc[te][cols]))
           for tr, te in KFold(5, shuffle=True, random_state=42).split(X)]
    loo = cross_val_predict(LinearRegression(), X[cols], y, cv=LeaveOneOut())
    splits = [r2_score(d, LinearRegression().fit(a[cols], c).predict(b[cols]))
              for a, b, c, d in (train_test_split(X, y, test_size=0.2, random_state=s) for s in range(200))]
    return {"R2 train": m.rsquared, "Adj R2 train": m.rsquared_adj, "AIC": m.aic, "BIC": m.bic,
            "R2 test (seed 42)": r2_score(y_te, p_te),
            "RMSE test": np.sqrt(mean_squared_error(y_te, p_te)), "MAE test": mean_absolute_error(y_te, p_te),
            "CV R2 (5-fold mean)": np.mean(cv5), "LOOCV R2": r2_score(y, loo),
            "Mean test R2 over 200 random splits": np.mean(splits)}, np.array(splits)


res_full, s_full = evaluate(X_COLS)
res_one, s_one = evaluate(["fifa_points_diff"])
comp = pd.DataFrame({"Full model (8 variables)": res_full, "FIFA points only (1 variable)": res_one}).round(3)
comp.loc["Share of 200 splits where full model wins"] = [round(np.mean(s_full > s_one), 2), np.nan]
comp.to_csv(f"{OUT}/table7_full_vs_fifa_only.csv")
print("\n", comp.to_string())

# ---------- 5. One compact coefficient table for the report (coef, SE, p, VIF) ----------
Xc = sm.add_constant(X_tr)
vif = [np.nan] + [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
tab = pd.DataFrame({"coef": full.params, "std_err": full.bse, "t": full.tvalues, "p": full.pvalues,
                    "std_beta": pd.concat([pd.Series({"const": np.nan}),
                                           full.params.drop("const") * X_tr.std() / y_tr.std()]),
                    "VIF": vif}).round(4)
tab.to_csv(f"{OUT}/table8_report_coefficients.csv")
print("\n", tab.to_string())

# ---------- 6. One figure with three diagnostic panels (saves report space) ----------
p_te = full.predict(sm.add_constant(X_te))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].scatter(full.fittedvalues, full.resid, alpha=0.7, s=18); ax[0].axhline(0, color="red", ls="--")
ax[0].set_xlabel("Fitted values"); ax[0].set_ylabel("Residuals"); ax[0].set_title("(a) Residuals vs fitted (train)")
sm.qqplot(full.resid, line="45", fit=True, ax=ax[1], markersize=4); ax[1].set_title("(b) Normal Q-Q of residuals")
ax[2].scatter(y_te, p_te, alpha=0.8, s=22); ax[2].plot([-5, 7], [-5, 7], "r--")
ax[2].set_xlim(-5, 7); ax[2].set_ylim(-5, 7)
ax[2].set_xlabel("Actual goal difference"); ax[2].set_ylabel("Predicted"); ax[2].set_title("(c) Actual vs predicted (test, n = 21)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_diagnostics_combined.png", dpi=200)
print(f"\nSaved tables 6-8 and fig7 in {OUT}/")
