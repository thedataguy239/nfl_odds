#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import requests
import nfl_framework as core
from data_ingestors import fetch_cfb_results_games, fetch_cfb_schedule_week


@dataclass
class LoopConfig:
    season: int
    week: Optional[int]
    db: Path
    outdir: Path
    hfa: float
    k: float
    k_playoff: float
    use_mov: bool


def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


def derive_current_week_from_results(results_df: pd.DataFrame) -> tuple[int, int]:
    if results_df.empty:
        return 0, 1
    wmax = int(results_df["week"].max())
    return wmax, wmax + 1


def run_weekly(cfg: LoopConfig) -> None:
    conn = core.connect(cfg.db)
    core.init_db(conn)
    results = fetch_cfb_results_games(cfg.season)
    engine = core.EloEngine(
        conn,
        core.EloConfig(k=cfg.k, k_playoff=cfg.k_playoff, hfa=cfg.hfa, use_mov=cfg.use_mov),
    )
    results_sorted = results.sort_values(["season", "date", "week"]).reset_index(drop=True)
    for _, r in results_sorted.iterrows():
        engine.update_ratings_from_game(
            season=int(r["season"]),
            week=int(r["week"]),
            date=str(r["date"]),
            home=str(r["home_team"]),
            away=str(r["away_team"]),
            home_score=int(r["home_score"]),
            away_score=int(r["away_score"]),
            neutral=int(r["neutral"]),
            playoff=int(r["playoff"]),
        )
    conn.commit()
    log(f"Applied {len(results_sorted)} finalized games to Elo ratings.")
    last_final, next_week = derive_current_week_from_results(results)
    target_week = cfg.week if (cfg.week and cfg.week != 0) else next_week
    log(f"Last finalized week: {last_final}; predicting week: {target_week}")
    sched = fetch_cfb_schedule_week(cfg.season, target_week)
    if sched.empty:
        log("No schedule found; nothing to predict.")
        return
    preds = []
    for _, g in sched.iterrows():
        gid, p_home = engine.predict_game(
            season=int(g["season"]),
            week=int(g["week"]),
            date=str(g["date"] or ""),
            home=str(g["home_team"]),
            away=str(g["away_team"]),
            neutral=int(g.get("neutral", 0)),
        )
        engine.save_prediction(
            gid,
            season=int(g["season"]),
            week=int(g["week"]),
            date=str(g["date"] or ""),
            home=str(g["home_team"]),
            away=str(g["away_team"]),
            p_home=p_home,
        )
        preds.append(
            {
                "game_id": gid,
                "season": int(g["season"]),
                "week": int(g["week"]),
                "date": str(g["date"] or ""),
                "home_team": str(g["home_team"]),
                "away_team": str(g["away_team"]),
                "p_home_win": float(p_home),
                "p_away_win": float(1.0 - p_home),
                "hfa": float(cfg.hfa),
                "k": float(cfg.k),
                "mov_enabled": int(cfg.use_mov),
            }
        )
    out_df = pd.DataFrame(preds)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.outdir / f"predictions_week{target_week}.csv"
    out_df.to_csv(out_path, index=False)
    conn.commit()
    log(f"Wrote {len(out_df)} predictions → {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Weekly CFB Elo automation loop (Power Five)")
    p.add_argument("--db", required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", default="auto")
    p.add_argument("--outdir", default="out/cfb_weekly")
    p.add_argument("--hfa", type=float, default=55.0)
    p.add_argument("--k", type=float, default=20.0)
    p.add_argument("--k-playoff", type=float, default=24.0)
    p.add_argument("--use-mov", action="store_true")
    return p


def parse_args(argv=None) -> LoopConfig:
    args = build_parser().parse_args(argv)
    week = None if isinstance(args.week, str) and args.week.lower() == "auto" else int(args.week)
    return LoopConfig(
        season=int(args.season),
        week=week,
        db=Path(args.db),
        outdir=Path(args.outdir),
        hfa=float(args.hfa),
        k=float(args.k),
        k_playoff=float(args.k_playoff),
        use_mov=bool(args.use_mov),
    )


def main(argv=None) -> None:
    cfg = parse_args(argv)
    try:
        run_weekly(cfg)
    except requests.HTTPError as e:
        log(f"HTTP error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
