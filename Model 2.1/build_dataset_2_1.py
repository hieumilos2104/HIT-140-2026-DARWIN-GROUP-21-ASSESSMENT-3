
import os
import numpy as np
import pandas as pd

# Raw files can be in a "source_data" sub-folder OR in the same folder as this script
RAW_DIR = "source_data" if os.path.isdir("source_data") else "."


def raw(name):
    return os.path.join(RAW_DIR, name)

HOSTS = {"USA", "Mexico", "Canada"}           # host nations named by FIFA
STRONG_CONF = {"UEFA", "CONMEBOL"}
FIRST_MATCH_REST = 7                           # same assumed rest for every team before its first match
NAME = {"Korea Republic": "South Korea", "Czechia": "Czech Republic", "Bosnia–Herz": "Bosnia & Herzegovina",
        "Türkiye": "Turkey", "Côte d'Ivoire": "Ivory Coast", "Cabo Verde": "Cape Verde",
        "IR Iran": "Iran", "Congo DR": "DR Congo"}


def load_fbref(path):
    fb = pd.read_csv(path).dropna(subset=["Round"]).reset_index(drop=True)
    fb["team_a"] = fb["Home"].map(lambda s: NAME.get(s.rsplit(" ", 1)[0], s.rsplit(" ", 1)[0]))
    fb["team_b"] = fb["Away"].map(lambda s: NAME.get(s.split(" ", 1)[1], s.split(" ", 1)[1]))
    g = fb["Score"].str.extract(r"(\d+)–(\d+)").astype(int)       # after extra time, shoot-out ignored
    fb["goals_a"], fb["goals_b"] = g[0], g[1]
    pens = fb["Score"].str.extract(r"^\((\d+)\).*\((\d+)\)$").astype(float)
    fb["pen_a"], fb["pen_b"] = pens[0], pens[1]
    fb["date"] = pd.to_datetime(fb["Date"])
    return fb


# ---------- World Cup 2022 stage reached (FBref 2022) ----------
# 0 = not at WC 2022, 1 = group stage, 2 = round of 16, 3 = quarter-final,
# 4 = fourth place, 5 = third place, 6 = runner-up, 7 = winner
wc22 = load_fbref(raw("fbref_wc2022_scores_fixtures.csv"))
# Prior for early-tournament form = WC 2022 average goals per team per match (172 goals / 128 = 1.344).
# One pseudo-match with this value smooths form before a team's first match.
PRIOR_GOALS = (wc22.goals_a.sum() + wc22.goals_b.sum()) / (2 * len(wc22))
stage22 = {t: 1 for t in pd.concat([wc22.team_a, wc22.team_b])}
for r in wc22.itertuples():
    a_wins = (r.goals_a, r.pen_a if r.pen_a == r.pen_a else 0) > (r.goals_b, r.pen_b if r.pen_b == r.pen_b else 0)
    winner, loser = (r.team_a, r.team_b) if a_wins else (r.team_b, r.team_a)
    if r.Round == "Round of 16":
        stage22[loser] = 2
    elif r.Round == "Quarter-finals":
        stage22[loser] = 3
    elif r.Round == "Third-place match":
        stage22[winner], stage22[loser] = 5, 4
    elif r.Round == "Final":
        stage22[winner], stage22[loser] = 7, 6

# ---------- World Cup 2026 results (FBref 2026) ----------
fb = load_fbref(raw("fbref_scores_fixtures.csv"))
fb["match_id"] = range(1, len(fb) + 1)

# per-team form from EARLIER 2026 matches only
long = pd.concat([fb.assign(team=fb.team_a, gf=fb.goals_a, ga=fb.goals_b),
                  fb.assign(team=fb.team_b, gf=fb.goals_b, ga=fb.goals_a)]).sort_values(["team", "date"])
grp = long.groupby("team")
n_prior = grp.cumcount()
long["prior_gf_pg"] = (grp["gf"].cumsum() - long["gf"] + PRIOR_GOALS) / (n_prior + 1)
long["prior_ga_pg"] = (grp["ga"].cumsum() - long["ga"] + PRIOR_GOALS) / (n_prior + 1)
long["rest_days"] = grp["date"].diff().dt.days.fillna(FIRST_MATCH_REST)
form = long.set_index(["match_id", "team"])

# ---------- FIFA ranking ----------
fifa = pd.read_csv(raw("fifa_ranking.csv")).set_index("team")
assert set(fb.team_a) | set(fb.team_b) <= set(fifa.index), "team missing from FIFA ranking"
fifa["trend_1y"] = fifa["fifa_points_jun2026"] - fifa["fifa_points_jul2025"]
fifa["strong_conf"] = fifa["confederation"].isin(STRONG_CONF).astype(int)


def team_diff(a, b, values):
    return values[a] - values[b]


def form_diff(r, col):
    return form.loc[(r.match_id, r.team_a), col] - form.loc[(r.match_id, r.team_b), col]


rows = []
for r in fb.itertuples():
    a, b = r.team_a, r.team_b
    rows.append({
        "match_id": r.match_id, "date": r.Date, "stage": r.Round, "team_a": a, "team_b": b,
        "fifa_points_diff": round(team_diff(a, b, fifa["fifa_points_jun2026"]), 2),
        "fifa_trend_1y_diff": round(team_diff(a, b, fifa["trend_1y"]), 2),
        "host_diff": int(a in HOSTS) - int(b in HOSTS),
        "rest_days_diff": form_diff(r, "rest_days"),
        "strong_conf_diff": team_diff(a, b, fifa["strong_conf"]),
        "wc2022_stage_diff": stage22.get(a, 0) - stage22.get(b, 0),
        "prior_gf_pg_diff": round(form_diff(r, "prior_gf_pg"), 4),
        "prior_ga_pg_diff": round(form_diff(r, "prior_ga_pg"), 4),
        "goal_diff": r.goals_a - r.goals_b,
    })
pd.DataFrame(rows).to_csv("model1_data.csv", index=False)
print("model1_data.csv written:", len(rows), "rows")
print("WC 2022 stage of 2026 teams:", {t: stage22.get(t, 0) for t in sorted(fifa.index) if stage22.get(t, 0) > 0})
