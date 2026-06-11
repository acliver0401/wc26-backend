"""
Match-result fetcher — resolves real (or realistically simulated) final scores
for completed matches.

In production, replace ``_simulate_result()`` with a call to a real sports-data
API (FlashScore / football-data.org / ESPN).  The simulator is biased toward the
stronger team but allows realistic upsets (~15 %).
"""

from __future__ import annotations

import json
import logging
import random
from datetime import date as date_type, timedelta
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_yesterday_results(
    reference_date: Optional[date_type] = None,
) -> list[dict]:
    """
    Return final results for every match scheduled on *yesterday* (relative
    to *reference_date*, default today).

    Each result dict::

        {
            "date": "2026-06-11",
            "home": "Mexico",
            "away": "South Africa",
            "home_score": 2,
            "away_score": 0,
            "outcome": "H",          # "H" | "D" | "A"
            "home_goals": 2,
            "away_goals": 0,
            "home_cards": 1,         # yellow + red
            "away_cards": 2,
        }
    """
    today = reference_date or date_type.today()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.isoformat()

    schedule = _load_schedule()
    played = [m for m in schedule if m["date"] <= yesterday_str]

    if not played:
        _logger.info("No completed matches found up to %s", yesterday_str)
        return []

    results: list[dict] = []
    for m in played:
        res = _simulate_result(m)
        results.append(res)

    _logger.info("Fetched %d results for matches through %s", len(results), yesterday_str)
    return results


def fetch_result_for_match(
    home: str, away: str, match_date: str,
) -> Optional[dict]:
    """Fetch / simulate a single match result."""
    schedule = _load_schedule()
    match = next(
        (m for m in schedule if m["home"] == home and m["away"] == away and m["date"] == match_date),
        None,
    )
    if match is None:
        return None
    return _simulate_result(match)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_schedule() -> list[dict]:
    path = DATA_DIR / "predictions.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_rankings() -> dict[str, int]:
    path = DATA_DIR / "fifa_rankings.json"
    with open(path, encoding="utf-8") as f:
        return {r["team"]: r["rank"] for r in json.load(f)}


def _simulate_result(match: dict) -> dict:
    """
    Generate a realistic match result biased by FIFA-rank gap.

    - Draw ~25 % of the time (real WC average).
    - The stronger side wins ~60 % of the time; weaker ~15 %.
    - Score distribution driven by Poisson with team-strength lambdas.
    """
    rankings = _load_rankings()
    home_rank = rankings.get(match["home"], 40)
    away_rank = rankings.get(match["away"], 40)

    rng = random.Random(hash(match["date"] + match["home"] + match["away"]) % (2**31))

    # --- Outcome roll -------------------------------------------------------
    rank_gap = abs(home_rank - away_rank)
    # Stronger side win prob
    stronger_win = 0.48 + rank_gap * 0.005  # 0.48 → ~0.68 for 40-rank gap
    draw_p = 0.25
    weaker_win = 1.0 - stronger_win - draw_p

    home_better = home_rank < away_rank
    roll = rng.random()
    if roll < draw_p:
        outcome = "D"
        home_goals = rng.choices([0, 1, 2, 3], weights=[30, 40, 20, 10])[0]
        away_goals = home_goals
    elif home_better:
        if roll < draw_p + stronger_win:
            outcome = "H"
            home_goals, away_goals = _score_pair(rng, stronger=True)
        else:
            outcome = "A"
            away_goals, home_goals = _score_pair(rng, stronger=True)
    else:
        if roll < draw_p + weaker_win:
            outcome = "A"
            away_goals, home_goals = _score_pair(rng, stronger=True)
        else:
            outcome = "H"
            home_goals, away_goals = _score_pair(rng, stronger=True)

    # Clamp to avoid impossible scores
    home_goals = max(0, min(home_goals, 6))
    away_goals = max(0, min(away_goals, 6))

    return {
        "date": match["date"],
        "home": match["home"],
        "away": match["away"],
        "outcome": outcome,
        "home_score": home_goals,
        "away_score": away_goals,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_cards": rng.choices([0, 1, 2, 3], weights=[30, 35, 25, 10])[0],
        "away_cards": rng.choices([0, 1, 2, 3, 4], weights=[20, 30, 30, 15, 5])[0],
    }


def _score_pair(rng: random.Random, stronger: bool) -> tuple[int, int]:
    """Return (stronger_goals, weaker_goals)."""
    if stronger:
        s = rng.choices([1, 2, 3, 4, 0], weights=[30, 30, 20, 10, 10])[0]
        w = rng.choices([0, 1, 2], weights=[50, 35, 15])[0]
    else:
        s = rng.choices([1, 2, 3], weights=[40, 40, 20])[0]
        w = rng.choices([0, 1, 2], weights=[40, 40, 20])[0]
    if s <= w:
        s = w + 1
    return s, w
