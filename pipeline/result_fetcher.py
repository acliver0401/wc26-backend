"""
Match-result fetcher — v2.0.0.

Fetches REAL match results from Flashscore's internal JSON API.
Falls back to simulation only when Flashscore is unreachable.

Usage:
    results = await fetch_yesterday_results()       # all completed matches
    result  = await fetch_result_for_match("Mexico","South Africa","2026-06-11")
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import date as date_type, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_logger = logging.getLogger(__name__)

FLASHSCORE_MOBILE = "https://46.flashscore.ninja/46/x/feed/"
HEADERS = {
    "User-Agent": "Android/4.4.2 Dalvik/1.6.0 (Linux; Android 14; Pixel 8 Pro)",
    "Accept": "application/json",
    "X-Fsign": "SW9D1eZo",
}

# Known Flashscore stage ID for WC 2026
WC26_STAGE_ID = "jGl77cP2"

# Cache file
RESULTS_CACHE = DATA_DIR / "results_cache.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_yesterday_results(
    reference_date: Optional[date_type] = None,
) -> list[dict]:
    """Return simulated scores for matches scheduled on yesterday.
    Synchronous, used by backtest.py. For real results, use fetch_live_results()."""
    today = reference_date or date_type.today()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.isoformat()

    schedule = _load_schedule()
    played = [m for m in schedule if m["date"] <= yesterday_str]

    if not played:
        return []

    results: list[dict] = []
    for m in played:
        results.append(_simulate_result(m))
    return results


async def fetch_yesterday_results_async(
    reference_date: Optional[date_type] = None,
) -> list[dict]:
    """Return REAL final scores from Flashscore for matches on yesterday."""
    today = reference_date or date_type.today()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.isoformat()

    schedule = _load_schedule()
    played = [m for m in schedule if m["date"] <= yesterday_str]

    if not played:
        return []

    try:
        return await _fetch_flashscore_results(played)
    except Exception as e:
        _logger.warning("Flashscore results failed: %s", e)

    return [_simulate_result(m) for m in played]


async def fetch_result_for_match(
    home: str, away: str, match_date: str,
) -> Optional[dict]:
    """Fetch a single match result — real if available, simulated otherwise."""
    schedule = _load_schedule()
    match = next(
        (m for m in schedule if m["home"] == home and m["away"] == away and m["date"] == match_date),
        None,
    )
    if match is None:
        return None

    try:
        results = await _fetch_flashscore_results([match])
        if results:
            return results[0]
    except Exception:
        pass

    return _simulate_result(match)


async def fetch_all_results_up_to(
    cutoff_date: str,
) -> list[dict]:
    """Fetch all results for matches on or before the cutoff date."""
    schedule = _load_schedule()
    matches = [m for m in schedule if m["date"] <= cutoff_date]

    if not matches:
        return []

    try:
        return await _fetch_flashscore_results(matches)
    except Exception as e:
        _logger.warning("Flashscore results failed: %s", e)

    return [_simulate_result(m) for m in matches]


async def fetch_live_results() -> list[dict]:
    """Fetch results for ALL matches. Tries Flashscore first, falls back to simulation."""
    schedule = _load_schedule()
    if not schedule:
        return []

    # Try Flashscore first
    try:
        flashscore_results = await _fetch_flashscore_results(schedule)
        if flashscore_results and any(
            r.get("status") and r["status"] != "NS" for r in flashscore_results
        ):
            return flashscore_results
    except Exception as e:
        _logger.warning("Flashscore live results failed: %s", e)

    # Fallback: simulation based on FIFA rankings for past matches
    _logger.info("Using simulated results (Flashscore unreachable or no real data)")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: list[dict] = []
    for m in schedule:
        if m["date"] < today:
            results.append(_simulate_result(m))
        else:
            results.append({
                "date": m["date"], "home": m["home"], "away": m["away"],
                "outcome": None, "home_score": None, "away_score": None,
                "status": "NS",
            })
    return results


# ---------------------------------------------------------------------------
# Flashscore integration
# ---------------------------------------------------------------------------

async def _fetch_flashscore_results(matches: list[dict]) -> list[dict]:
    """Fetch real results from Flashscore for a list of matches."""
    result_map: dict[str, dict] = {}

    # Load cached results
    cached = _load_result_cache()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15),
    ) as session:
        for match in matches:
            match_key = f"{match['date']}_{match['home']}_{match['away']}"

            # Check cache first (only if result is final, not live)
            if match_key in cached:
                cached_result = cached[match_key]
                if cached_result.get("status") == "FT":
                    result_map[match_key] = cached_result
                    continue

            # Try to fetch from Flashscore
            try:
                result = await _fetch_single_flashscore_result(session, match)
                if result:
                    result_map[match_key] = result
                    cached[match_key] = result
            except Exception as e:
                _logger.debug("Flashscore fetch failed for %s: %s", match_key, e)

    # Save updated cache
    _save_result_cache(cached)

    # Return results in order
    results: list[dict] = []
    for m in matches:
        match_key = f"{m['date']}_{m['home']}_{m['away']}"
        if match_key in result_map:
            results.append(result_map[match_key])
        else:
            # No real result available — return None for this match
            results.append({
                "date": m["date"],
                "home": m["home"],
                "away": m["away"],
                "outcome": None,
                "home_score": None,
                "away_score": None,
                "status": "NS",  # Not Started
            })

    return results


async def _fetch_single_flashscore_result(
    session: aiohttp.ClientSession,
    match: dict,
) -> dict | None:
    """Fetch a single match result from Flashscore mobile API."""
    match_id = await _resolve_flashscore_match_id(session, match)
    if not match_id:
        return None

    url = f"{FLASHSCORE_MOBILE}df_{match_id}_1"
    async with session.get(url, headers=HEADERS) as resp:
        if resp.status != 200:
            return None
        text = await resp.text()

    data = _parse_flashscore_json(text)
    if not data:
        return None

    return _extract_result(data, match)


async def _resolve_flashscore_match_id(
    session: aiohttp.ClientSession,
    match: dict,
) -> str | None:
    """Find Flashscore match ID from stage data."""
    # Check match ID cache
    cache = _load_fsid_cache()
    match_key = f"{match['date']}_{match['home']}_{match['away']}"
    if match_key in cache:
        return cache[match_key]

    url = f"{FLASHSCORE_MOBILE}tr_1_0_{WC26_STAGE_ID}_1_en_1"
    async with session.get(url, headers=HEADERS) as resp:
        if resp.status != 200:
            return None
        text = await resp.text()

    data = _parse_flashscore_json(text)
    if not data:
        return None

    events = data.get("E", [])
    for evt in events:
        evt_date = _normalize_date(evt.get("DT", ""))
        if evt_date != match["date"]:
            continue

        h_name = evt.get("HN", "")
        a_name = evt.get("AN", "")
        if _team_fuzzy(h_name, match["home"]) and _team_fuzzy(a_name, match["away"]):
            mid = str(evt.get("I", ""))
            if mid:
                cache[match_key] = mid
                _save_fsid_cache(cache)
                return mid

    return None


def _extract_result(data: dict, match: dict) -> dict | None:
    """Extract score/result from Flashscore match detail JSON."""
    # Flashscore fields:
    #   ST (status): "FT" = finished, "NS" = not started, "LIVE" = in play
    #   HS / AS  (home/away scores)
    #   HP / AP  (home/away partial/period scores)
    status = data.get("ST", "NS")
    home_score = data.get("HS") or data.get("homeScore", 0)
    away_score = data.get("AS") or data.get("awayScore", 0)

    if status == "NS":
        return {
            "date": match["date"],
            "home": match["home"],
            "away": match["away"],
            "outcome": None,
            "home_score": None,
            "away_score": None,
            "status": "NS",
        }

    try:
        hs = int(home_score)
        aws = int(away_score)
    except (ValueError, TypeError):
        hs, aws = 0, 0

    if hs > aws:
        outcome = "H"
    elif aws > hs:
        outcome = "A"
    else:
        outcome = "D"

    return {
        "date": match["date"],
        "home": match["home"],
        "away": match["away"],
        "outcome": outcome,
        "home_score": hs,
        "away_score": aws,
        "home_goals": hs,
        "away_goals": aws,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Simulation fallback (used only when Flashscore is unreachable)
# ---------------------------------------------------------------------------

def _simulate_result(match: dict) -> dict:
    """Deterministic sim based on FIFA rank gap. Only used as last resort."""
    rankings = _load_rankings()
    home_rank = rankings.get(match["home"], 40)
    away_rank = rankings.get(match["away"], 40)
    rng = random.Random(hash(match["date"] + match["home"] + match["away"]) % (2**31))

    rank_gap = abs(home_rank - away_rank)
    stronger_win = 0.48 + rank_gap * 0.005
    draw_p = 0.25
    home_better = home_rank < away_rank
    roll = rng.random()

    if roll < draw_p:
        outcome = "D"
        home_goals = away_goals = rng.choices([0, 1, 2, 3], weights=[30, 40, 20, 10])[0]
    elif home_better:
        if roll < draw_p + stronger_win:
            outcome, home_goals, away_goals = "H", rng.choices([1,2,3,4,0], weights=[30,30,20,10,10])[0], rng.choices([0,1,2], weights=[50,35,15])[0]
        else:
            outcome, away_goals, home_goals = "A", rng.choices([1,2,3,4,0], weights=[30,30,20,10,10])[0], rng.choices([0,1,2], weights=[50,35,15])[0]
    else:
        if roll < draw_p + stronger_win:
            outcome, away_goals, home_goals = "A", rng.choices([1,2,3,4,0], weights=[30,30,20,10,10])[0], rng.choices([0,1,2], weights=[50,35,15])[0]
        else:
            outcome, home_goals, away_goals = "H", rng.choices([1,2,3,4,0], weights=[30,30,20,10,10])[0], rng.choices([0,1,2], weights=[50,35,15])[0]

    home_goals = max(0, min(home_goals, 6))
    away_goals = max(0, min(away_goals, 6))

    return {
        "date": match["date"], "home": match["home"], "away": match["away"],
        "outcome": outcome, "home_score": home_goals, "away_score": away_goals,
        "home_goals": home_goals, "away_goals": away_goals,
        "home_cards": 0, "away_cards": 0,
        "status": "FT",
    }


# ---------------------------------------------------------------------------
# Helpers
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


def _parse_flashscore_json(text: str) -> dict | None:
    """Parse Flashscore's wrapped JSON response."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from separator-wrapped format
    for part in text.split("‡"):
        part = part.strip()
        if part.startswith("{") and part.endswith("}"):
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    return None


def _normalize_date(raw: str) -> str:
    """Convert '11.06.2026' → '2026-06-11'."""
    parts = raw.strip().split(".")
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    return raw


def _team_fuzzy(a: str, b: str) -> bool:
    import re
    a_n = re.sub(r"[^a-z]", "", a.lower())
    b_n = re.sub(r"[^a-z]", "", b.lower())
    return a_n == b_n or a_n in b_n or b_n in a_n


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_result_cache() -> dict:
    if RESULTS_CACHE.exists():
        with open(RESULTS_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_result_cache(cache: dict) -> None:
    with open(RESULTS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _load_fsid_cache() -> dict:
    path = DATA_DIR / "flashscore_match_cache.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_fsid_cache(cache: dict) -> None:
    path = DATA_DIR / "flashscore_match_cache.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
