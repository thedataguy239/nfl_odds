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

## Local (optional)
```bash
python nfl_framework.py init-db --db nfl.db
python nfl_framework.py new-season --db nfl.db --season 2025
python weekly_loop.py --db nfl.db --season 2025 --week auto --outdir out/weekly --hfa 65 --k 20 --use-mov
python blended_predictor.py --db nfl.db --season 2025 --week auto --outdir out/weekly
python compare_odds.py --season 2025 --week 2 --blended out/weekly/predictions_blended_week2.csv --outdir out/weekly --threshold 0.15
```
