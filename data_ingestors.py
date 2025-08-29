from __future__ import annotations
import pandas as pd, numpy as np, requests
from typing import Dict, List
NFL_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/games/games.csv.gz"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
CFB_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/college-football/scoreboard"
import pandas as pd, numpy as np, requests
from cfb_teams import POWER_FIVE_TEAMS
import nfl_data_py as nfl

TEAM_NORMALIZE = {"WSH": "WAS", "LA": "LAR", "OAK": "LV", "SD": "LAC"}

def fetch_results_games(season: int) -> pd.DataFrame:
    df = nfl.import_schedules([season])
    df = df[(df["season"] == season) & df["home_score"].notna() & df["away_score"].notna()].copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["REG", "REGULAR"])]
    for col in ["home_team", "away_team"]:
        df[col] = df[col].replace(TEAM_NORMALIZE)
    date_series = pd.to_datetime(df.get("gameday", df.get("start_time")))
    neutral_series = df["neutral_site"] if "neutral_site" in df.columns else pd.Series(0, index=df.index)
    
    out = pd.DataFrame({
        "season": df["season"].astype(int),
        "week": df["week"].astype(int),
        "date": date_series.dt.date.astype(str),
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_score": df["home_score"].astype(int),
        "away_score": df["away_score"].astype(int),
        "neutral": neutral_series.fillna(0).astype(int),
        "playoff": 0,
    })
    if "game_type" in df.columns:
        out["playoff"] = np.where(df["game_type"].isin(["WC","DIV","CON","SB","P","POST"]), 1, 0).astype(int)

        out["playoff"] = np.where(df["game_type"].isin(["WC","DIV","CON","SB","P","POST"]), 1, 0).astype(int)
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


def fetch_cfb_results_games(season: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for week in range(1, 16):
        params = {"week": week, "seasontype": 2, "dates": season}
        r = requests.get(CFB_SCOREBOARD, params=params, timeout=20)
        data = r.json()
        for ev in data.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            teams = comp.get("competitors", [])
            home = next((t for t in teams if t.get("homeAway") == "home"), None)
            away = next((t for t in teams if t.get("homeAway") == "away"), None)
            if not home or not away: continue
            h_abbr = (home.get("team", {}) or {}).get("abbreviation")
            a_abbr = (away.get("team", {}) or {}).get("abbreviation")
            if h_abbr not in POWER_FIVE_TEAMS or a_abbr not in POWER_FIVE_TEAMS: continue
            date_iso = comp.get("date")
            date_str = pd.to_datetime(date_iso).tz_convert("UTC").date().isoformat() if date_iso else ""
            rows.append({
                "season": season,
                "week": week,
                "date": date_str,
                "home_team": h_abbr,
                "away_team": a_abbr,
                "home_score": int(home.get("score") or 0),
                "away_score": int(away.get("score") or 0),
                "neutral": int(comp.get("neutralSite") or 0),
                "playoff": 0,
            })
    return pd.DataFrame(rows)

def fetch_cfb_schedule_week(season: int, week: int) -> pd.DataFrame:
    params = {"week": week, "seasontype": 2, "dates": season}
    r = requests.get(CFB_SCOREBOARD, params=params, timeout=20)
    data = r.json(); rows: List[Dict] = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]; date_iso = comp.get("date")
        date_str = pd.to_datetime(date_iso).tz_convert("UTC").date().isoformat() if date_iso else ""
        neutral = int(comp.get("neutralSite") or 0); teams = comp.get("competitors", [])
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away: continue
        h_abbr = (home.get("team", {}) or {}).get("abbreviation")
        a_abbr = (away.get("team", {}) or {}).get("abbreviation")
        if h_abbr not in POWER_FIVE_TEAMS or a_abbr not in POWER_FIVE_TEAMS: continue
        rows.append({"season": season, "week": week, "date": date_str,
                     "home_team": h_abbr,
                     "away_team": a_abbr,
                     "neutral": neutral})
    return pd.DataFrame(rows)
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
