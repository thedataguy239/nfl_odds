#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import requests, pandas as pd, numpy as np
import nfl_framework as core
NFLFASTR_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/games/games.csv.gz"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
TEAM_NORMALIZE = {"WSH": "WAS", "LA": "LAR", "OAK": "LV", "SD": "LAC"}
@dataclass
class LoopConfig:
    season: int; week: Optional[int]; db: Path; outdir: Path; hfa: float; k: float; k_playoff: float; use_mov: bool
def log(msg: str):
    ts = datetime.now().isoformat(timespec="seconds"); print(f"[{ts}] {msg}")
import nfl_data_py as nfl  # NEW

TEAM_NORMALIZE = {"WSH": "WAS", "LA": "LAR", "OAK": "LV", "SD": "LAC"}

def fetch_nfl_results(season: int) -> pd.DataFrame:
    """Load final game results for a season using nfl_data_py (schedules + scores)."""
    log("Loading games via nfl_data_py.import_schedules…")
    df = nfl.import_schedules([season])

    # Filter to this season + games with final scores
    df = df[(df["season"] == season) & df["home_score"].notna() & df["away_score"].notna()].copy()

    # Date column hygiene
if "gameday" in df.columns:
    date_series = pd.to_datetime(df["gameday"])
else:
    date_series = pd.to_datetime(df.get("start_time", pd.NaT))

# Neutral-site column (Series of zeros if missing)
neutral_series = df["neutral_site"] if "neutral_site" in df.columns else pd.Series(0, index=df.index)

df_out = pd.DataFrame({
    "season": df["season"].astype(int),
    "week": df["week"].astype(int),
    "date": date_series.dt.date.astype(str),
    "home_team": df["home_team"].astype(str),
    "away_team": df["away_team"].astype(str),
    "home_score": df["home_score"].astype(int),
    "away_score": df["away_score"].astype(int),
    "neutral": neutral_series.fillna(0).astype(int),
    "playoff": 0,
})

# Mark playoffs if present
if "game_type" in df.columns:
    df_out["playoff"] = np.where(df["game_type"].isin(["WC", "DIV", "CON", "SB", "P", "POST"]), 1, 0).astype(int)

    return df_out

def fetch_espn_schedule(season: int, week: int) -> pd.DataFrame:
    params = {"week": week, "seasontype": 2, "dates": season}
    log(f"Fetching ESPN schedule for season={season}, week={week}…")
    resp = requests.get(ESPN_SCOREBOARD, params=params, timeout=20); resp.raise_for_status(); data = resp.json()
    rows: List[Dict] = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]; date_iso = comp.get("date")
        date_str = pd.to_datetime(date_iso).tz_convert("UTC").date().isoformat() if date_iso else ""
        neutral = int(comp.get("neutralSite") or 0); teams = comp.get("competitors", [])
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away: continue
        def abbr(n): return (n.get("team", {}) or {}).get("abbreviation")
        rows.append({"season": int(season), "week": int(week), "date": date_str,
                     "home_team": abbr(home), "away_team": abbr(away), "neutral": neutral})
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["home_team", "away_team"]: df[col] = df[col].replace(TEAM_NORMALIZE)
    return df
def derive_current_week_from_results(results_df: pd.DataFrame):
    if results_df.empty: return 0, 1
    wmax = int(results_df["week"].max()); return wmax, wmax + 1
def run_weekly(cfg: LoopConfig) -> None:
    conn = core.connect(cfg.db); core.init_db(conn)
    results = fetch_nfl_results(cfg.season)
    engine = core.EloEngine(conn, core.EloConfig(k=cfg.k, k_playoff=cfg.k_playoff, hfa=cfg.hfa, use_mov=cfg.use_mov))
    results_sorted = results.sort_values(["season", "date", "week"]).reset_index(drop=True)
    for _, r in results_sorted.iterrows():
        engine.update_ratings_from_game(season=int(r["season"]), week=int(r["week"]), date=str(r["date"]),
                                        home=str(r["home_team"]), away=str(r["away_team"]),
                                        home_score=int(r["home_score"]), away_score=int(r["away_score"]),
                                        neutral=int(r["neutral"]), playoff=int(r["playoff"]))
    conn.commit(); log(f"Applied {len(results_sorted)} finalized games to Elo ratings.")
    last_final, next_week = derive_current_week_from_results(results)
    target_week = cfg.week if (cfg.week and cfg.week != 0) else next_week
    log(f"Last finalized week: {last_final}; predicting week: {target_week}")
    sched = fetch_espn_schedule(cfg.season, target_week)
    if sched.empty: log("No schedule found; nothing to predict."); return
    preds = []
    for _, g in sched.iterrows():
        gid, p_home = engine.predict_game(season=int(g["season"]), week=int(g["week"]), date=str(g["date"] or ""),
                                          home=str(g["home_team"]), away=str(g["away_team"]), neutral=int(g.get("neutral", 0)))
        engine.save_prediction(gid, season=int(g["season"]), week=int(g["week"]), date=str(g["date"] or ""),
                               home=str(g["home_team"]), away=str(g["away_team"]), p_home=p_home)
        preds.append({"game_id": gid, "season": int(g["season"]), "week": int(g["week"]), "date": str(g["date"] or ""),
                      "home_team": str(g["home_team"]), "away_team": str(g["away_team"]),
                      "p_home_win": float(p_home), "p_away_win": float(1.0 - p_home),
                      "hfa": float(cfg.hfa), "k": float(cfg.k), "mov_enabled": int(cfg.use_mov)})
    out_df = pd.DataFrame(preds); cfg.outdir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.outdir / f"predictions_week{target_week}.csv"; out_df.to_csv(out_path, index=False); conn.commit()
    log(f"Wrote {len(out_df)} predictions → {out_path}")
def build_parser():
    import argparse
    p = argparse.ArgumentParser(description="Weekly NFL Elo automation loop")
    p.add_argument("--db", required=True); p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", default="auto"); p.add_argument("--outdir", default="out/weekly")
    p.add_argument("--hfa", type=float, default=65.0); p.add_argument("--k", type=float, default=20.0)
    p.add_argument("--k-playoff", type=float, default=24.0); p.add_argument("--use-mov", action="store_true")
    return p
def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    week = None if isinstance(args.week, str) and args.week.lower() == "auto" else int(args.week)
    return LoopConfig(season=int(args.season), week=week, db=Path(args.db), outdir=Path(args.outdir),
                      hfa=float(args.hfa), k=float(args.k), k_playoff=float(args.k_playoff), use_mov=bool(args.use_mov))
def main(argv=None):
    cfg = parse_args(argv)
    try: run_weekly(cfg)
    except requests.HTTPError as e: log(f"HTTP error: {e}"); sys.exit(2)
if __name__ == "__main__": main()
