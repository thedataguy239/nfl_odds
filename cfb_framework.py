#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Iterable

from nfl_framework import (
    connect,
    init_db,
    EloConfig,
    EloEngine,
    cmd_ingest_results,
    cmd_predict_week,
    cmd_backtest,
    cmd_tune_elo,
)
from cfb_teams import POWER_FIVE_TEAMS


def cmd_init_db(args):
    conn = connect(Path(args.db))
    init_db(conn)
    conn.commit()
    print(f"Initialized DB at {args.db}")


def cmd_new_season(args):
    conn = connect(Path(args.db))
    init_db(conn)
    cfg = EloConfig()
    engine = EloEngine(conn, cfg)
    for team in sorted(POWER_FIVE_TEAMS):
        engine.upsert_rating(team, int(args.season), cfg.base_rating)
    conn.commit()
    print(f"Initialized {len(POWER_FIVE_TEAMS)} teams for season {args.season} at rating {cfg.base_rating}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="College Football Elo framework (Power Five)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init-db", help="Create SQLite tables")
    s.add_argument("--db", required=True)
    s.set_defaults(func=cmd_init_db)

    s = sub.add_parser("new-season", help="Initialize team ratings for a season")
    s.add_argument("--db", required=True)
    s.add_argument("--season", required=True, type=int)
    s.set_defaults(func=cmd_new_season)

    s = sub.add_parser("ingest-results", help="Ingest a results CSV and update Elo ratings")
    s.add_argument("--db", required=True)
    s.add_argument("--csv", required=True)
    s.add_argument("--k", type=float, default=20.0)
    s.add_argument("--k-playoff", type=float, default=24.0)
    s.add_argument("--hfa", type=float, default=55.0)
    s.add_argument("--use-mov", action="store_true")
    s.add_argument("--columns-json")
    s.set_defaults(func=cmd_ingest_results)

    s = sub.add_parser("predict-week", help="Predict a week's games from a schedule CSV")
    s.add_argument("--db", required=True)
    s.add_argument("--schedule", required=True)
    s.add_argument("--season", type=int)
    s.add_argument("--week", type=int)
    s.add_argument("--k", type=float, default=20.0)
    s.add_argument("--k-playoff", type=float, default=24.0)
    s.add_argument("--hfa", type=float, default=55.0)
    s.add_argument("--use-mov", action="store_true")
    s.add_argument("--columns-json")
    s.add_argument("--out")
    s.set_defaults(func=cmd_predict_week)

    s = sub.add_parser("backtest", help="Backtest Elo parameters on a historical results CSV")
    s.add_argument("--csv", required=True)
    s.add_argument("--k", type=float, default=20.0)
    s.add_argument("--k-playoff", type=float, default=24.0)
    s.add_argument("--hfa", type=float, default=55.0)
    s.add_argument("--use-mov", action="store_true")
    s.add_argument("--columns-json")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("tune-elo", help="Grid search K/HFA/MOV to minimize log loss")
    s.add_argument("--csv", required=True)
    s.add_argument("--columns-json")
    s.set_defaults(func=cmd_tune_elo)

    return p


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
