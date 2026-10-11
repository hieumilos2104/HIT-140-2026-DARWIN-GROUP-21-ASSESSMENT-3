HIT140 - Objective 2 - Linear Regression 2.1
Predicting goal difference in FIFA World Cup 2026 matches
=========================================================

FILES
  source_data/          Raw data copied from the websites and saved as CSV:
    fbref_scores_fixtures.csv          FBref, 2026 World Cup Scores & Fixtures (104 matches)
    fbref_wc2022_scores_fixtures.csv   FBref, 2022 World Cup Scores & Fixtures (64 matches)
    fifa_ranking.csv                   FIFA Men's World Ranking - points on 11 Jun 2026 and
                                       July 2025, confederation of each team (48 teams)
  build_dataset_2_1.py  Data wrangling in Python: calculates the 8 variables and goal_diff
  model1_data.csv       Modelling dataset: 104 rows (1 row = 1 actual World Cup 2026 match)
  model_2_1.py          Python code: EDA, OLS model, assumption checks, evaluation

HOW TO RUN
  pip install pandas numpy matplotlib statsmodels scipy scikit-learn
  python build_dataset_2_1.py      (step 1: raw data -> model1_data.csv)
  python model_2_1.py              (step 2: analysis and model)
  Results (tables, charts, results.txt) are saved in the folder outputs_2.1/

DATASET
  ID columns (not explanatory variables): match_id, date, stage, team_a, team_b
  Team A = home team as listed by FBref; Team B = away team.
  Target variable: goal_diff = goals of Team A - goals of Team B
                   (extra time included, penalty shoot-outs excluded)
  8 explanatory variables (Team A minus Team B; positive = advantage for Team A):
    1. fifa_points_diff     FIFA ranking points, 11 Jun 2026                   (current strength)
    2. fifa_trend_1y_diff   Change in FIFA points, July 2025 -> 11 Jun 2026   (one-year trend)
    3. host_diff            +1 if Team A is a host nation (USA, Mexico, Canada),
                            -1 if Team B is, 0 otherwise                       (home advantage)
    4. rest_days_diff       Days since previous World Cup match                (fatigue)
    5. strong_conf_diff     1 if the team is in UEFA or CONMEBOL, else 0       (regional strength)
    6. wc2022_stage_diff    Stage reached at World Cup 2022 (0 = absent,
                            1 = group ... 7 = winner)                          (recent tournament pedigree)
    7. prior_gf_pg_diff     Goals scored per game in earlier 2026 matches      (attacking form)
    8. prior_ga_pg_diff     Goals conceded per game in earlier 2026 matches    (defensive form)

PRE-MATCH INFORMATION
  Variables 1, 2, 3, 5, 6 are known before the tournament
  (the 2026 FIFA ranking was published hours before the opening match).
  Variables 4, 7, 8 use only matches played before the match being predicted
  (first match of every team: rest = 7 days; form smoothed with one pseudo-match of 1.4 goals,
   so form differences are 0 on matchday 1). No in-match statistics are used.

DATA SOURCES
  FBref, https://fbref.com/en/
    - 2026 FIFA World Cup Scores & Fixtures -> goal_diff and variables 4, 7, 8
    - 2022 FIFA World Cup Scores & Fixtures -> variable 6
  FIFA official website, https://inside.fifa.com/fifa-world-ranking/men
    - FIFA/Coca-Cola Men's World Ranking (11 Jun 2026, July 2025) and confederations
      -> variables 1, 2, 5; host nations -> variable 3
