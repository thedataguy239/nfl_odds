from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from odds_ingestor import odds_with_implied_probs

DEF_THRESH = 0.15


def main(season: int, week: int, blended_csv: str, outdir: str, threshold: float = DEF_THRESH):
    path = Path(blended_csv)
    if not path.is_file():
        print(f"Blended predictions file not found: {blended_csv}")
        return

    preds = pd.read_csv(path)
    odds = odds_with_implied_probs(season, week)
    if preds.empty or odds.empty:
        print("Missing inputs; preds or odds empty.")
        return
    merged = preds.merge(odds, on=["home_team","away_team"], how="inner")
    p_model = merged["p_home_win_blended"].astype(float) if "p_home_win_blended" in merged.columns else merged["p_home_win"].astype(float)
    p_market = merged["p_home_impl"].astype(float).fillna(0.5)
    delta = p_model - p_market; merged["model_vs_market"] = delta; merged["abs_diff"] = np.abs(delta)
    outliers = merged[merged["abs_diff"] >= float(threshold)].copy()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out_path = Path(outdir) / f"outliers_week{week}.csv"; outliers.sort_values("abs_diff", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote {len(outliers)} outliers → {out_path}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--season", type=int, required=True); ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--blended", required=True); ap.add_argument("--outdir", default="out/weekly"); ap.add_argument("--threshold", type=float, default=DEF_THRESH)
    args = ap.parse_args(); main(season=args.season, week=args.week, blended_csv=args.blended, outdir=args.outdir, threshold=args.threshold)
