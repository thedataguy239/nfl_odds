# NFL Predictor (Elo + Recent Form + Injuries + Odds Outliers)

Weekly, fully-online NFL win probabilities using GitHub Actions. Produces:
- `out/weekly/predictions_weekN.csv` (Elo)
- `out/weekly/predictions_blended_weekN.csv` (Elo + recent-form + injuries)
- `out/weekly/outliers_weekN.csv` (model vs market differences)
- `nfl.db` (SQLite ratings persistence)

## GitHub Actions (no local setup)
1. Create a new repo on GitHub.
2. Upload everything from this ZIP to the repo root.
3. Go to **Actions** → enable the workflow.
4. It runs every **Tuesday 9:00 AM ET**; artifacts attach to each run.

Optional: set **Actions Secret** `ODDS_API_KEY` for broader odds coverage.

## College Football (Power Five only)
Free ESPN data can power similar workflows for college football. The helpers
`fetch_cfb_results_games`, `fetch_cfb_schedule_week`, and
`odds_with_implied_probs_cfb` filter to games where both teams are from Power
Five conferences. A parallel Elo loop lives in `weekly_loop_cfb.py`, and the
companion CLI `cfb_framework.py` initializes ratings for all Power Five teams.

## Local (optional)
```bash
python nfl_framework.py init-db --db nfl.db
python nfl_framework.py new-season --db nfl.db --season 2025
python weekly_loop.py --db nfl.db --season 2025 --week auto --outdir out/weekly --hfa 65 --k 20 --use-mov
python blended_predictor.py --db nfl.db --season 2025 --week auto --outdir out/weekly
python compare_odds.py --season 2025 --week 2 --blended out/weekly/predictions_blended_week2.csv --outdir out/weekly --threshold 0.15
```

### College Football example
```bash
python cfb_framework.py init-db --db cfb.db
python cfb_framework.py new-season --db cfb.db --season 2025
python weekly_loop_cfb.py --db cfb.db --season 2025 --week auto --outdir out/cfb_weekly --hfa 55 --k 20 --use-mov
```
