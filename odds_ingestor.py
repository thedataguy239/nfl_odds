from __future__ import annotations
import os, requests, pandas as pd
from typing import Dict, List
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
ODDS_API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
TEAM_NORMALIZE = {"WSH": "WAS", "LA": "LAR", "OAK": "LV", "SD": "LAC"}
def american_to_prob(american):
    if american is None or american == 0: return None
    a = float(american); return 100.0/(a+100.0) if a>0 else (-a)/((-a)+100.0)
def implied_from_two(p_home, p_away):
    if p_home is None or p_away is None: return {"home": p_home, "away": p_away}
    s = p_home + p_away; 
    if s == 0: return {"home": None, "away": None}
    return {"home": p_home/s, "away": p_away/s}
def fetch_espn_odds(season: int, week: int) -> pd.DataFrame:
    params = {"week": week, "seasontype": 2, "dates": season}
    r = requests.get(ESPN_SCOREBOARD, params=params, timeout=20); data = r.json()
    rows: List[Dict] = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]; teams = comp.get("competitors", [])
        if len(teams) != 2: continue
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away: continue
        odds_arr = comp.get("odds") or []; moneyline_home = moneyline_away = None; spread = None; book = None
        if odds_arr:
            o0 = odds_arr[0]; book = (o0.get("provider") or {}).get("name"); details = o0.get("details") or ""
            if "homeTeamOdds" in o0 and "awayTeamOdds" in o0:
                try: moneyline_home = float((o0["homeTeamOdds"] or {}).get("moneyLine", None))
                except: pass
                try: moneyline_away = float((o0["awayTeamOdds"] or {}).get("moneyLine", None))
                except: pass
            if isinstance(details, str) and details.strip():
                try: spread = float(details.split()[-1])
                except: pass
        rows.append({"home_team": (home.get("team", {}) or {}).get("abbreviation"),
                     "away_team": (away.get("team", {}) or {}).get("abbreviation"),
                     "home_ml": moneyline_home, "away_ml": moneyline_away, "spread": spread,
                     "book": book or "ESPN", "source": "ESPN"})
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["home_team","away_team"]: df[col] = df[col].replace(TEAM_NORMALIZE)
    return df
def fetch_odds_api() -> pd.DataFrame:
    key = os.getenv("ODDS_API_KEY")
    if not key: return pd.DataFrame()
    params = {"apiKey": key, "regions": "us", "markets": "h2h,spreads", "oddsFormat": "american", "dateFormat": "iso"}
    r = requests.get(ODDS_API, params=params, timeout=20)
    if r.status_code != 200: return pd.DataFrame()
    data = r.json(); rows: List[Dict] = []
    for game in data:
        home = (game.get("home_team") or "").upper(); away = (game.get("away_team") or "").upper()
        best_home_ml = best_away_ml = None; best_spread = None
        for b in game.get("bookmakers") or []:
            for mk in b.get("markets") or []:
                if mk.get("key") == "h2h":
                    for o in mk.get("outcomes") or []:
                        name = (o.get("name") or "").upper(); price = o.get("price")
                        if name.startswith(home[:3]): best_home_ml = price if (best_home_ml is None) else max(best_home_ml, price)
                        elif name.startswith(away[:3]): best_away_ml = price if (best_away_ml is None) else max(best_away_ml, price)
                elif mk.get("key") == "spreads":
                    for o in mk.get("outcomes") or []:
                        if (o.get("name") or "").upper().startswith(home[:3]): best_spread = o.get("point", best_spread)
        rows.append({"home_team": home[:3].replace("GIA","NYG"), "away_team": away[:3].replace("JET","NYJ"),
                     "home_ml": best_home_ml, "away_ml": best_away_ml, "spread": best_spread, "book": "BestAvailable", "source": "OddsAPI"})
    return pd.DataFrame(rows)
def odds_with_implied_probs(season: int, week: int) -> pd.DataFrame:
    import pandas as pd
    espn = fetch_espn_odds(season, week); oddsapi = fetch_odds_api()
    df = pd.concat([espn, oddsapi], ignore_index=True)
    if df.empty: return df
    df["p_home_ml"] = df["home_ml"].apply(american_to_prob); df["p_away_ml"] = df["away_ml"].apply(american_to_prob)
    imp = df.apply(lambda r: implied_from_two(r["p_home_ml"], r["p_away_ml"]), axis=1, result_type="expand")
    df["p_home_impl"] = imp["home"]; df["p_away_impl"] = imp["away"]
    df["pref"] = df["source"].apply(lambda s: 1 if s == "OddsAPI" else 0)
    df = df.sort_values(["home_team","away_team","pref"], ascending=[True,True,False])
    df = df.drop_duplicates(subset=["home_team","away_team"], keep="first").drop(columns=["pref"])
    return df
