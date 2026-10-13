"""
HIT140 Objective 2 - Linear Regression 2.1 (supplement)
Variable-selection evidence and a compact figure for the report.

Run AFTER model_2_1.py (uses the same data, split and random_state).
Input : model1_data.csv
Output: outputs_2.1/table6_selection_steps.txt
        outputs_2.1/table7_full_vs_fifa_only.csv
        outputs_2.1/table8_report_coefficients.csv
        outputs_2.1/fig7_diagnostics_combined.png   (two-panel Figure 1 for the report)
        outputs_2.1/table9_report_checks.txt   (further checks quoted in the report)
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

# ---------- 6. One compact two-panel figure for the report ----------
p_te = full.predict(sm.add_constant(X_te))
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].scatter(full.fittedvalues, full.resid, alpha=0.7, s=16); ax[0].axhline(0, color="red", ls="--")
ax[0].set_xlabel("Fitted values"); ax[0].set_ylabel("Residuals"); ax[0].set_title("(a) Residuals vs fitted (training)", fontsize=10)
ax[1].scatter(y_te, p_te, alpha=0.8, s=20); ax[1].plot([-5, 7], [-5, 7], "r--")
ax[1].set_xlim(-5, 7); ax[1].set_ylim(-5, 7)
ax[1].set_xlabel("Actual goal difference"); ax[1].set_ylabel("Predicted"); ax[1].set_title("(b) Actual vs predicted (test, n = 21)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_diagnostics_combined.png", dpi=200)
print(f"\nSaved tables 6-8 and fig7 in {OUT}/")

# ---------- 7. Further checks quoted in the report ----------
from scipy import stats
from statsmodels.stats.diagnostic import linear_reset

chk = []
# data facts
knockout = df["stage"] != "Group stage"
chk.append(f"Knockout matches level after extra time (decided on penalties): {int((knockout & (y == 0)).sum())}")
chk.append(f"Matches involving a host nation (host_diff != 0): {int((df['host_diff'] != 0).sum())}")
chk.append(f"Rest-day difference within -1..+1: {int(df['rest_days_diff'].between(-1, 1).sum())} of {len(df)} matches")
appear = pd.concat([df[["match_id", "date", "team_a"]].rename(columns={"team_a": "team"}),
                    df[["match_id", "date", "team_b"]].rename(columns={"team_b": "team"})])
appear = appear.sort_values(["date", "match_id"])
appear["n_prior"] = appear.groupby("team").cumcount()
n_prior = appear.groupby("match_id")["n_prior"].first()
chk.append(f"Earlier 2026 matches per team before a match: min {n_prior.min()}, median {int(n_prior.median())}, max {n_prior.max()}")
tt = stats.ttest_1samp(df["fifa_points_diff"], 0)
chk.append(f"Team A has more FIFA points in {np.mean(df['fifa_points_diff'] > 0):.1%} of matches; "
           f"mean gap = {df['fifa_points_diff'].mean():.1f} points, one-sample t({len(df) - 1}) = {tt.statistic:.2f}, p = {tt.pvalue:.3f}")

# assumption checks beyond model_2_1.py
reset = linear_reset(full, power=2, use_f=True)
chk.append(f"Ramsey RESET (linearity): F({int(reset.df_num)}, {int(reset.df_denom)}) = {float(reset.fvalue):.2f}, p = {float(reset.pvalue):.3f}")
cooks = full.get_influence().cooks_distance[0]
chk.append(f"Largest Cook's distance: {cooks.max():.3f} (matches above 0.5: {int((cooks > 0.5).sum())})")
hc3 = full.get_robustcov_results(cov_type="HC3")
chk.append(f"fifa_points_diff p with HC3 robust standard errors: {hc3.pvalues[list(full.params.index).index('fifa_points_diff')]:.3f}")
order = sorted(full.resid.index, key=lambda i: (df.loc[i, "date"], df.loc[i, "match_id"]))
chk.append(f"Durbin-Watson: {sm.stats.durbin_watson(full.resid):.2f} in shuffled training order, "
           f"{sm.stats.durbin_watson(full.resid.loc[order]):.2f} in date order")

# how typical is the seed-42 test set?
chk.append(f"SD of goal difference: training {y_tr.std():.2f}, test {y_te.std():.2f}, all {y.std():.2f}")
chk.append(f"Seed-42 test R2 ({res_full['R2 test (seed 42)']:.3f}) is higher than {np.mean(s_full < res_full['R2 test (seed 42)']):.0%} "
           f"of the 200 random splits")

# direction of the result (leave-one-out, decisive matches only)
loo_full = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
decisive = y != 0
chk.append(f"Host-nation matches in the training set: {int((X_tr['host_diff'] != 0).sum())}")
chk.append(f"Mean test R2 over 200 random splits: {s_full.mean():.4f}; single splits range from {s_full.min():.2f} to {s_full.max():.2f}")
chk.append(f"Picking the team with more FIFA points: correct winner in "
           f"{np.mean(np.sign(df['fifa_points_diff'][y != 0]) == np.sign(y[y != 0])):.1%} of decisive matches")
chk.append(f"Leave-one-out: correct winner in {np.mean(np.sign(loo_full[decisive]) == np.sign(y[decisive])):.1%} "
           f"of {int(decisive.sum())} decisive matches; always picking Team A: {np.mean(y[decisive] > 0):.1%}")

with open(f"{OUT}/table9_report_checks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chk))
print("\nFURTHER CHECKS\n" + "\n".join(chk))
