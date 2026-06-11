"""
Mock injury & team-status data generator.

In production this would scrape sites like Transfermarkt, Flashscore, or
call a paid API (Sportmonks / API-Football).  For now it produces
realistic-looking data so the full prediction pipeline can run.

Output structure per team:
    {
      "team": "Argentina",
      "core_injuries": 1,
      "total_absent": 2,
      "form_index": 0.72,          // 0-1, recent competitive form
      "form_trend": "up",          // up | stable | down
      "fatigue_index": 0.35,       // 0-1, accumulated minutes in last 14 days
      "key_player_doubtful": false,
      "updated_at": "2026-06-11T06:00:00Z"
    }
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Fixed seed so mock values are stable across calls within the same run,
# but will drift slightly each scheduler tick.
_rng = random.Random()


def _load_json(name: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Realistic per-team base values (hand-curated to mirror 2026 expectations)
# ---------------------------------------------------------------------------

# Teams with known injury concerns heading into 2026
_CORE_INJURY_BASE: dict[str, int] = {
    "Argentina": 0, "Brazil": 1, "France": 2, "Spain": 1, "England": 1,
    "Germany": 2, "Portugal": 1, "Netherlands": 0, "Belgium": 2,
    "Uruguay": 0, "Croatia": 1, "Colombia": 0, "Morocco": 1, "Mexico": 1,
    "Japan": 0, "United States": 2, "Senegal": 0, "South Korea": 1,
    "Switzerland": 0, "Austria": 1,
}

_FORM_BASE: dict[str, float] = {
    "Argentina": 0.82, "Brazil": 0.78, "France": 0.75, "Spain": 0.80,
    "England": 0.73, "Germany": 0.68, "Portugal": 0.76, "Netherlands": 0.72,
    "Belgium": 0.65, "Uruguay": 0.74, "Croatia": 0.66, "Colombia": 0.70,
    "Morocco": 0.71, "Mexico": 0.72, "Japan": 0.69, "United States": 0.64,
    "Senegal": 0.68, "South Korea": 0.63, "Switzerland": 0.67, "Austria": 0.65,
}

_FORM_TREND: dict[str, str] = {
    "Argentina": "up", "Brazil": "stable", "France": "down", "Spain": "up",
    "England": "stable", "Germany": "up", "Portugal": "up",
    "Netherlands": "stable", "Belgium": "down", "Uruguay": "up",
    "Croatia": "stable", "Colombia": "up", "Morocco": "up",
    "Mexico": "stable", "Japan": "up", "United States": "down",
    "Senegal": "stable", "South Korea": "down", "Switzerland": "stable",
    "Austria": "up",
}


def _drift(value: float, max_delta: float = 0.06) -> float:
    """Add a small random drift to a base value, clamped to [0, 1]."""
    delta = _rng.uniform(-max_delta, max_delta)
    return round(max(0.0, min(1.0, value + delta)), 2)


def generate_injuries(seed: int | None = None) -> list[dict]:
    """
    Generate (or refresh) injury & form data for all 48 teams.

    Pass ``seed`` for deterministic output (e.g. ``seed=hash(datetime.now())``).
    """
    if seed is not None:
        _rng.seed(seed)

    rankings = _load_json("fifa_rankings.json")
    now = datetime.utcnow().isoformat() + "Z"
    results: list[dict] = []

    for r in rankings:
        team = r["team"]
        base_injuries = _CORE_INJURY_BASE.get(team, _rng.randint(0, 2))
        # occasionally a key player becomes doubtful
        key_doubtful = _rng.random() < 0.12

        form = _FORM_BASE.get(team, _rng.uniform(0.45, 0.75))
        trend = _FORM_TREND.get(team, _rng.choice(["up", "stable", "down"]))

        # Fatigue index: higher for teams that played recent qualifiers / friendlies
        fatigue = _drift(_rng.uniform(0.20, 0.55), 0.08)

        total_absent = base_injuries + (1 if key_doubtful else 0)

        results.append({
            "team": team,
            "core_injuries": base_injuries,
            "total_absent": total_absent,
            "form_index": round(form, 2),
            "form_trend": trend,
            "fatigue_index": fatigue,
            "key_player_doubtful": key_doubtful,
            "updated_at": now,
        })

    # Sort by team name for readability
    results.sort(key=lambda x: x["team"])

    cache_path = DATA_DIR / "injury_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {"updated_at": now, "teams": results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return results


def load_injury_cache() -> dict | None:
    """Return cached injury data, or None."""
    cache_path = DATA_DIR / "injury_cache.json"
    if not cache_path.exists():
        return None
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def get_injuries_for_team(team: str) -> dict | None:
    """Convenience: return injury data for a single team."""
    cache = load_injury_cache()
    if cache is None:
        return None
    for t in cache.get("teams", []):
        if t["team"] == team:
            return t
    return None
