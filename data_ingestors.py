from __future__ import annotations
import pandas as pd, numpy as np, requests
from typing import Dict, List
NFL_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/games/games.csv.gz"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
TEAM_NORMALIZE = {"WSH": "WAS", "LA": "LAR", "OAK": "LV", "SD": "LAC"}
def fetch_results_games(season: int) -> pd.DataFrame:
    df = pd.read_csv(NFL_GAMES_URL, compression="infer"); df = df[df["season"] == season].copy()
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    for col in ["home_team", "away_team"]: df[col] = df[col].replace(TEAM_NORMALIZE)
    out = pd.DataFrame({"season": df["season"].astype(int), "week": df["week"].astype(int),
                        "date": pd.to_datetime(df.get("game_date", df.get("gameday"))).dt.date.astype(str),
                        "home_team": df["home_team"], "away_team": df["away_team"],
                        "home_score": df["home_score"].astype(int), "away_score": df["away_score"].astype(int),
                        "neutral": df.get("neutral_site", 0).fillna(0).astype(int),
                        "playoff": np.where(df["game_type"].fillna("").isin(["WC","DIV","CON","SB","P"]), 1, 0)})
    return out
def fetch_schedule_week(season: int, week: int) -> pd.DataFrame:
    params = {"week": week, "seasontype": 2, "dates": season}; r = requests.get(ESPN_SCOREBOARD, params=params, timeout=20)
    data = r.json(); rows: List[Dict] = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]; date_iso = comp.get("date")
        date_str = pd.to_datetime(date_iso).tz_convert("UTC").date().isoformat() if date_iso else ""
        neutral = int(comp.get("neutralSite") or 0); teams = comp.get("competitors", [])
        home = next((t for t in teams if t.get("homeAway") == "home"), None); away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away: continue
        rows.append({"season": season, "week": week, "date": date_str,
                     "home_team": (home.get("team", {}) or {}).get("abbreviation"),
                     "away_team": (away.get("team", {}) or {}).get("abbreviation"),
                     "neutral": neutral})
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["home_team", "away_team"]: df[col] = df[col].replace(TEAM_NORMALIZE)
    return df
def recent_form_feature(results: pd.DataFrame, window_games: int = 4) -> pd.DataFrame:
    long_rows = []
    for _, r in results.iterrows():
        long_rows.append({"team": r["home_team"], "date": r["date"], "pd": int(r["home_score"]) - int(r["away_score"])})
        long_rows.append({"team": r["away_team"], "date": r["date"], "pd": int(r["away_score"]) - int(r["home_score"])})
    long = pd.DataFrame(long_rows); long["date"] = pd.to_datetime(long["date"]); long = long.sort_values(["team","date"])
    def roll_pd(g):
        g = g.copy(); g["recent_pd_pg"] = g["pd"].rolling(window_games, min_periods=1).mean().shift(1); return g
    feat = long.groupby("team", group_keys=False).apply(roll_pd)
    feat = feat.dropna(subset=["recent_pd_pg"]); return feat[["team","date","recent_pd_pg"]]
