#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
DDL = {
    "ratings": ("CREATE TABLE IF NOT EXISTS ratings (\n"
                "  team TEXT NOT NULL,\n"
                "  season INTEGER NOT NULL,\n"
                "  rating REAL NOT NULL,\n"
                "  updated_at TEXT NOT NULL,\n"
                "  PRIMARY KEY(team, season)\n"
                ")"),
    "games": ("CREATE TABLE IF NOT EXISTS games (\n"
              "  game_id TEXT NOT NULL,\n"
              "  season INTEGER NOT NULL,\n"
              "  week INTEGER NOT NULL,\n"
              "  date TEXT NOT NULL,\n"
              "  home_team TEXT NOT NULL,\n"
              "  away_team TEXT NOT NULL,\n"
              "  home_score INTEGER,\n"
              "  away_score INTEGER,\n"
              "  neutral INTEGER DEFAULT 0,\n"
              "  playoff INTEGER DEFAULT 0,\n"
              "  PRIMARY KEY(game_id)\n"
              ")"),
    "predictions": ("CREATE TABLE IF NOT EXISTS predictions (\n"
                    "  game_id TEXT NOT NULL,\n"
                    "  season INTEGER NOT NULL,\n"
                    "  week INTEGER NOT NULL,\n"
                    "  date TEXT NOT NULL,\n"
                    "  home_team TEXT NOT NULL,\n"
                    "  away_team TEXT NOT NULL,\n"
                    "  p_home_win REAL NOT NULL,\n"
                    "  hfa REAL NOT NULL,\n"
                    "  k REAL NOT NULL,\n"
                    "  mov_enabled INTEGER NOT NULL,\n"
                    "  created_at TEXT NOT NULL,\n"
                    "  PRIMARY KEY(game_id)\n"
                    ")"),
}
def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn
def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for sql in DDL.values():
        cur.execute(sql)
    conn.commit()
@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k: float = 20.0
    k_playoff: float = 24.0
    hfa: float = 65.0
    use_mov: bool = True
    def k_for_game(self, playoff: bool) -> float:
        return self.k_playoff if playoff else self.k
def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))
def mov_multiplier(margin: int, rating_diff: float) -> float:
    if margin <= 0: return 1.0
    return math.log(max(1, margin) + 1.0) * (2.2 / ((abs(rating_diff) * 0.001) + 2.2))
class EloEngine:
    def __init__(self, conn: sqlite3.Connection, cfg: EloConfig):
        self.conn = conn; self.cfg = cfg
    def get_rating(self, team: str, season: int) -> float:
        cur = self.conn.cursor()
        cur.execute("SELECT rating FROM ratings WHERE team=? AND season=?", (team, season))
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT rating FROM ratings WHERE team=? AND season=?", (team, season - 1))
            row_prev = cur.fetchone()
            base = row_prev[0] if row_prev else self.cfg.base_rating
            self.upsert_rating(team, season, base); return base
        return float(row[0])
    def upsert_rating(self, team: str, season: int, rating: float) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO ratings(team, season, rating, updated_at) VALUES(?,?,?,?)\n"
            "ON CONFLICT(team, season) DO UPDATE SET rating=excluded.rating, updated_at=excluded.updated_at",
            (team, season, float(rating), now),
        )
    def _game_id(self, season: int, week: int, home: str, away: str, date: str) -> str:
        return f"{season}_{week}_{home}_vs_{away}_{date}"

    def record_game(self, *, season: int, week: int, date: str, home: str, away: str,
                    home_score: Optional[int], away_score: Optional[int], neutral: int, playoff: int) -> str:
        gid = self._game_id(season, week, home, away, date)
        self.conn.execute(
            "INSERT OR REPLACE INTO games(game_id, season, week, date, home_team, away_team, home_score, away_score, neutral, playoff)\n"
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (gid, season, week, date, home, away, home_score, away_score, neutral, playoff),
        ); return gid
    def update_ratings_from_game(self, *, season: int, week: int, date: str, home: str, away: str,
                                 home_score: int, away_score: int, neutral: int, playoff: int) -> None:
        r_home = self.get_rating(home, season); r_away = self.get_rating(away, season)
        hfa = 0.0 if neutral else self.cfg.hfa
        exp_home = expected_score(r_home + hfa, r_away)
        if home_score == away_score: s_home, margin = 0.5, 0
        else: s_home, margin = (1.0 if home_score > away_score else 0.0), abs(home_score - away_score)
        k = self.cfg.k_for_game(bool(playoff))
        mult = mov_multiplier(margin, (r_home + hfa) - r_away) if self.cfg.use_mov else 1.0
        delta = k * mult * (s_home - exp_home)
        self.upsert_rating(home, season, r_home + delta)
        self.upsert_rating(away, season, r_away - delta)
        self.record_game(season=season, week=week, date=date, home=home, away=away,
                         home_score=home_score, away_score=away_score, neutral=int(neutral), playoff=int(playoff))
    def predict_game(self, *, season: int, week: int, date: str, home: str, away: str, neutral: int) -> Tuple[str, float]:
        r_home = self.get_rating(home, season); r_away = self.get_rating(away, season)
        hfa = 0.0 if neutral else self.cfg.hfa
        p_home = expected_score(r_home + hfa, r_away)
        gid = self._game_id(season, week, home, away, date); return gid, float(p_home)
    def save_prediction(self, gid: str, *, season: int, week: int, date: str, home: str, away: str, p_home: float) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO predictions(game_id, season, week, date, home_team, away_team, p_home_win, hfa, k, mov_enabled, created_at)\n"
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (gid, season, week, date, home, away, float(p_home), self.cfg.hfa, self.cfg.k, int(self.cfg.use_mov), now),
        )
RESULTS_DEFAULTS = {"season":"season","week":"week","date":"date","home_team":"home_team","away_team":"away_team",
                    "home_score":"home_score","away_score":"away_score","neutral":"neutral","playoff":"playoff"}
SCHEDULE_DEFAULTS = {"season":"season","week":"week","date":"date","home_team":"home_team","away_team":"away_team","neutral":"neutral"}
def normalize_bool(x) -> int:
    import pandas as pd, numpy as np
    if pd.isna(x): return 0
    if isinstance(x, (int, np.integer)): return int(x)
    s = str(x).strip().lower(); return 1 if s in {"1","true","t","yes","y"} else 0
def cmd_init_db(args):
    conn = connect(Path(args.db)); init_db(conn); conn.commit(); print(f"Initialized DB at {args.db}")
def cmd_new_season(args):
    conn = connect(Path(args.db)); init_db(conn); cfg = EloConfig(); engine = EloEngine(conn, cfg)
    if args.teams_csv:
        df = pd.read_csv(args.teams_csv); col = args.teams_col or "team"
        teams = df[col].dropna().astype(str).str.strip().unique()
    else:
        teams = ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"]
    for team in teams: engine.upsert_rating(str(team), int(args.season), cfg.base_rating)
    conn.commit(); print(f"Initialized {len(list(teams))} teams for season {args.season} at rating {cfg.base_rating}")
def cmd_ingest_results(args):
    conn = connect(Path(args.db)); init_db(conn)
    cfg = EloConfig(k=args.k, k_playoff=args.k_playoff, hfa=args.hfa, use_mov=bool(args.use_mov))
    engine = EloEngine(conn, cfg); cols = {**RESULTS_DEFAULTS}
    if args.columns_json: cols.update(json.loads(args.columns_json))
    df = pd.read_csv(args.csv); df[cols["date"]] = pd.to_datetime(df[cols["date"]])
    df = df.sort_values([cols["season"], cols["date"], cols["week"]])
    for _, row in df.iterrows():
        season, week = int(row[cols["season"]]), int(row[cols["week"]])
        date = pd.to_datetime(row[cols["date"]]).date().isoformat()
        home, away = str(row[cols["home_team"]]).strip(), str(row[cols["away_team"]]).strip()
        hsc, asc = int(row[cols["home_score"]]), int(row[cols["away_score"]])
        neutral = normalize_bool(row.get(cols["neutral"], 0)); playoff = normalize_bool(row.get(cols["playoff"], 0))
        engine.update_ratings_from_game(season=season, week=week, date=date, home=home, away=away,
                                        home_score=hsc, away_score=asc, neutral=neutral, playoff=playoff)
    conn.commit(); print(f"Ingested {len(df)} games and updated ratings.")
def cmd_predict_week(args):
    conn = connect(Path(args.db)); init_db(conn)
    cfg = EloConfig(k=args.k, k_playoff=args.k_playoff, hfa=args.hfa, use_mov=bool(args.use_mov)); engine = EloEngine(conn, cfg)
    cols = {**SCHEDULE_DEFAULTS}; if args.columns_json: cols.update(json.loads(args.columns_json))
    df = pd.read_csv(args.schedule)
    if args.season is not None: df = df[df[cols["season"]] == int(args.season)]
    if args.week is not None: df = df[df[cols["week"]] == int(args.week)]
    df[cols["date"]] = pd.to_datetime(df[cols["date"]]).dt.date.astype(str)
    out_rows = []
    for _, row in df.iterrows():
        season, week = int(row[cols["season"]]), int(row[cols["week"]])
        date = str(row[cols["date"]]); home, away = str(row[cols["home_team"]]).strip(), str(row[cols["away_team"]]).strip()
        neutral = normalize_bool(row.get(cols["neutral"], 0))
        gid, p_home = engine.predict_game(season=season, week=week, date=date, home=home, away=away, neutral=neutral)
        engine.save_prediction(gid, season=season, week=week, date=date, home=home, away=away, p_home=p_home)
        out_rows.append({"game_id": gid,"season": season,"week": week,"date": date,"home_team": home,"away_team": away,
                         "p_home_win": p_home,"p_away_win": 1.0 - p_home,"hfa": cfg.hfa,"k": cfg.k,"mov_enabled": int(cfg.use_mov)})
    out_df = pd.DataFrame(out_rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); out_df.to_csv(args.out, index=False)
        print(f"Wrote {len(out_df)} predictions to {args.out}")
    else: print(out_df.to_string(index=False))
    conn.commit()
def _iter_games(df: pd.DataFrame, cols: Dict[str, str]):
    df = df.copy(); df[cols["date"]] = pd.to_datetime(df[cols["date"]])
    df = df.sort_values([cols["season"], cols["date"], cols["week"]])
    for _, row in df.iterrows():
        yield {"season": int(row[cols["season"]]), "week": int(row[cols["week"]]), "date": row[cols["date"]].date().isoformat(),
               "home": str(row[cols["home_team"]]).strip(), "away": str(row[cols["away_team"]]).strip(),
               "home_score": int(row[cols["home_score"]]), "away_score": int(row[cols["away_score"]]),
               "neutral": normalize_bool(row.get(cols["neutral"], 0)), "playoff": normalize_bool(row.get(cols["playoff"], 0))}
def cmd_backtest(args):
    cols = {**RESULTS_DEFAULTS}; 
    if args.columns_json: cols.update(json.loads(args.columns_json))
    df = pd.read_csv(args.csv); conn = sqlite3.connect(":memory:"); init_db(conn)
    cfg = EloConfig(k=args.k, k_playoff=args.k_playoff, hfa=args.hfa, use_mov=bool(args.use_mov)); engine = EloEngine(conn, cfg)
    y_true, y_prob, n_games = [], [], 0
    for g in _iter_games(df, cols):
        _, p_home = engine.predict_game(season=g["season"], week=g["week"], date=g["date"], home=g["home"], away=g["away"], neutral=g["neutral"])
        y_true.append(0.5 if g["home_score"] == g["away_score"] else (1.0 if g["home_score"] > g["away_score"] else 0.0))
        y_prob.append(p_home)
        engine.update_ratings_from_game(**{"season":g["season"],"week":g["week"],"date":g["date"],"home":g["home"],"away":g["away"],
                                           "home_score":g["home_score"],"away_score":g["away_score"],"neutral":g["neutral"],"playoff":g["playoff"]})
        n_games += 1
    probs = np.clip(np.array(y_prob), 1e-6, 1-1e-6); y = np.array(y_true)
    preds = (probs >= 0.5).astype(float); acc = float((preds == (y == 1.0)).mean())
    brier = float(np.mean((probs - y) ** 2)); ll = float(log_loss(y, probs))
    print(json.dumps({"n_games": n_games,"accuracy@0.5": round(acc,4),"brier": round(brier,6),"log_loss": round(ll,6),
                      "k": cfg.k,"k_playoff": cfg.k_playoff,"hfa": cfg.hfa,"use_mov": int(cfg.use_mov)}, indent=2))
def cmd_tune_elo(args):
    cols = {**RESULTS_DEFAULTS}; 
    if args.columns_json: cols.update(json.loads(args.columns_json))
    df = pd.read_csv(args.csv); k_grid = [12,16,20,24,28]; hfa_grid = [50,60,65,70,80]; use_mov_opts = [0,1]; results = []
    for k in k_grid:
        for hfa in hfa_grid:
            for use_mov in use_mov_opts:
                conn = sqlite3.connect(":memory:"); init_db(conn)
                cfg = EloConfig(k=float(k), hfa=float(hfa), use_mov=bool(use_mov)); engine = EloEngine(conn, cfg)
                y_true, y_prob = [], []
                for g in _iter_games(df, cols):
                    _, p_home = engine.predict_game(season=g["season"], week=g["week"], date=g["date"], home=g["home"], away=g["away"], neutral=g["neutral"])
                    y_true.append(0.5 if g["home_score"] == g["away_score"] else (1.0 if g["home_score"] > g["away_score"] else 0.0))
                    y_prob.append(p_home)
                    engine.update_ratings_from_game(**{"season":g["season"],"week":g["week"],"date":g["date"],"home":g["home"],"away":g["away"],
                                                       "home_score":g["home_score"],"away_score":g["away_score"],"neutral":g["neutral"],"playoff":g["playoff"]})
                probs = np.clip(np.array(y_prob), 1e-6, 1 - 1e-6); y = np.array(y_true)
                from sklearn.metrics import log_loss as _ll; ll = float(_ll(y, probs)); brier = float(np.mean((probs - y) ** 2))
                results.append({"k": k, "hfa": hfa, "use_mov": use_mov, "log_loss": ll, "brier": brier})
    res_df = pd.DataFrame(results).sort_values(["log_loss", "brier"]).reset_index(drop=True)
    print("\nTuning results (best first):\n"); print(res_df.head(15).to_string(index=False))
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NFL Game Winner Framework — Elo-based core"); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init-db", help="Create SQLite tables"); s.add_argument("--db", required=True); s.set_defaults(func=cmd_init_db)
    s = sub.add_parser("new-season", help="Initialize team ratings for a season"); s.add_argument("--db", required=True); s.add_argument("--season", required=True, type=int)
    s.add_argument("--teams-csv"); s.add_argument("--teams-col"); s.set_defaults(func=cmd_new_season)
    s = sub.add_parser("ingest-results", help="Ingest a results CSV and update Elo ratings")
    s.add_argument("--db", required=True); s.add_argument("--csv", required=True); s.add_argument("--k", type=float, default=20.0)
    s.add_argument("--k-playoff", type=float, default=24.0); s.add_argument("--hfa", type=float, default=65.0); s.add_argument("--use-mov", action="store_true")
    s.add_argument("--columns-json"); s.set_defaults(func=cmd_ingest_results)
    s = sub.add_parser("predict-week", help="Predict a week's games from a schedule CSV")
    s.add_argument("--db", required=True); s.add_argument("--schedule", required=True); s.add_argument("--season", type=int); s.add_argument("--week", type=int)
    s.add_argument("--k", type=float, default=20.0); s.add_argument("--k-playoff", type=float, default=24.0); s.add_argument("--hfa", type=float, default=65.0)
    s.add_argument("--use-mov", action="store_true"); s.add_argument("--columns-json"); s.add_argument("--out"); s.set_defaults(func=cmd_predict_week)
    s = sub.add_parser("backtest", help="Backtest Elo parameters on a historical results CSV")
    s.add_argument("--csv", required=True); s.add_argument("--k", type=float, default=20.0); s.add_argument("--k-playoff", type=float, default=24.0)
    s.add_argument("--hfa", type=float, default=65.0); s.add_argument("--use-mov", action="store_true"); s.add_argument("--columns-json"); s.set_defaults(func=cmd_backtest)
    s = sub.add_parser("tune-elo", help="Grid search K/HFA/MOV to minimize log loss")
    s.add_argument("--csv", required=True); s.add_argument("--columns-json"); s.set_defaults(func=cmd_tune_elo); return p
def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser(); args = parser.parse_args(argv); args.func(args)
if __name__ == "__main__": main()
