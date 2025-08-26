from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from nfl_framework import EloConfig, EloEngine, connect, init_db
from data_ingestors import fetch_results_games, fetch_schedule_week, recent_form_feature
from injuries import matchup_injury_delta
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
W_ELO = 1.00; W_FORM = 0.020; W_INJ = 0.015; HFA_POINTS = 65.0
def derive_weeks(results: pd.DataFrame) -> int:
    return 1 if results.empty else int(results["week"].max() + 1)
def blended_week(db: str, season: int, week: int | None, outdir: str):
    conn = connect(Path(db)); init_db(conn); engine = EloEngine(conn, EloConfig(hfa=HFA_POINTS))
    res = fetch_results_games(season); next_week = derive_weeks(res); target_week = week or next_week
    form = recent_form_feature(res, window_games=4); sched = fetch_schedule_week(season, target_week)
    if sched.empty: print("No schedule to predict."); return
    rows = []
    for _, g in sched.iterrows():
        home, away, date = g["home_team"], g["away_team"], g["date"]
        gid, p_home = engine.predict_game(season=season, week=target_week, date=date, home=home, away=away, neutral=int(g.get("neutral", 0)))
        p = float(np.clip(p_home, 1e-6, 1-1e-6)); elo_logit = float(np.log(p/(1-p)))
        date_dt = pd.to_datetime(date)
        f_home = form[(form.team == home) & (form.date <= date_dt)].tail(1)
        f_away = form[(form.team == away) & (form.date <= date_dt)].tail(1)
        home_form = float(f_home.recent_pd_pg.iloc[0]) if len(f_home) else 0.0
        away_form = float(f_away.recent_pd_pg.iloc[0]) if len(f_away) else 0.0
        form_delta = home_form - away_form
        inj_delta = -matchup_injury_delta(home, away)
        blended_logit = (W_ELO * elo_logit) + (W_FORM * form_delta) + (W_INJ * inj_delta)
        p_blended = float(sigmoid(blended_logit))
        rows.append({"game_id": gid, "season": season, "week": target_week, "date": date,
                     "home_team": home, "away_team": away,
                     "p_home_win_elo": float(p_home), "p_home_win_blended": p_blended,
                     "recent_form_delta": round(form_delta, 3), "injury_delta": round(inj_delta, 3)})
    out = pd.DataFrame(rows); Path(outdir).mkdir(parents=True, exist_ok=True)
    out_path = Path(outdir) / f"predictions_blended_week{target_week}.csv"; out.to_csv(out_path, index=False)
    print(f"Wrote blended predictions → {out_path}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--week", default="auto"); ap.add_argument("--outdir", default="out/weekly"); args = ap.parse_args()
    week = None if str(args.week).lower() == "auto" else int(args.week); blended_week(args.db, args.season, week, args.outdir)
