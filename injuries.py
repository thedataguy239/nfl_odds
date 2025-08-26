from __future__ import annotations
import requests
from teams_map import TEAMS
ESPN_INJ_URL = "https://site.api.espn.com/apis/v2/sports/football/nfl/teams/{team_id}/injuries"
POS_WEIGHTS = {"QB": 40,"LT": 18,"RT": 18,"C": 12,"LG": 12,"RG": 12,"WR": 8,"TE": 6,"RB": 4,"CB": 10,"S": 6,"LB": 6,"EDGE": 8,"DT": 8}
STATUS_MULT = {"Out": 1.00,"Doubtful": 0.75,"Questionable": 0.40,"Suspension": 0.80}
def team_injury_penalty(abbr: str) -> float:
    meta = TEAMS.get(abbr); if not meta: return 0.0
    url = ESPN_INJ_URL.format(team_id=meta["espn_id"]); r = requests.get(url, timeout=20); data = r.json(); total = 0.0
    for group in (data.get("injuries") or []):
        for item in group.get("injuries") or []:
            pos = (item.get("position") or {}).get("abbreviation") or (item.get("athlete", {}) or {}).get("position", {}).get("abbreviation")
            status = (item.get("status") or {}).get("type", {}).get("name") or item.get("status", {}).get("name")
            total += POS_WEIGHTS.get(pos, 0) * STATUS_MULT.get(status, 0)
    return float(min(total, 70.0))
def matchup_injury_delta(home_abbr: str, away_abbr: str) -> float:
    home_pen = team_injury_penalty(home_abbr); away_pen = team_injury_penalty(away_abbr); return float(away_pen - home_pen)
