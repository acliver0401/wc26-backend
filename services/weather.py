"""
Live weather fetching via Open-Meteo (free, no API key required).

For each of the 16 World Cup 2026 stadiums, fetches:
  - daily max/min temperature (°C)
  - max relative humidity (%)
  - max precipitation probability (%)

Results are cached to ``data/weather_cache.json`` so the scheduler, predictor,
and API can all consume the latest snapshot without re-fetching.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Open-Meteo archive + forecast endpoint (no API key needed)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Parameters we request
WEATHER_PARAMS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "relative_humidity_2m_max",
]


def _load_stadiums() -> list[dict]:
    with open(DATA_DIR / "stadium_meta.json", encoding="utf-8") as f:
        return json.load(f)


def _match_date_for_stadium(stadium: dict, match_dates: list[str]) -> Optional[str]:
    """Return the first match date scheduled at this stadium."""
    sid = stadium["id"]
    for m in match_dates:
        if m["stadium_id"] == sid:
            return m["date"]
    return None


async def fetch_weather_for_stadium(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    target_date: str,
) -> dict | None:
    """
    Fetch the weather forecast for a single stadium on a given date.

    Open-Meteo returns daily data; we pick the day matching ``target_date``.
    """
    try:
        r = await client.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lng,
                "daily": ",".join(WEATHER_PARAMS),
                "timezone": "auto",
                "start_date": target_date,
                "end_date": target_date,
            },
            timeout=15.0,
        )
        r.raise_for_status()
        body = r.json()
        daily = body.get("daily", {})
        if not daily or "time" not in daily or len(daily["time"]) == 0:
            logger.warning("Open-Meteo returned empty daily for %.4f,%.4f on %s", lat, lng, target_date)
            return None

        idx = 0
        return {
            "date": daily["time"][idx],
            "temp_max_c": daily.get("temperature_2m_max", [None])[idx],
            "temp_min_c": daily.get("temperature_2m_min", [None])[idx],
            "precip_prob_max_pct": daily.get("precipitation_probability_max", [0])[idx] or 0,
            "humidity_max_pct": daily.get("relative_humidity_2m_max", [None])[idx],
        }
    except Exception:
        logger.exception("Weather fetch failed for %.4f,%.4f", lat, lng)
        return None


async def fetch_all_weather(
    match_schedule: list[dict] | None = None,
) -> list[dict]:
    """
    Fetch current weather forecasts for all 16 stadiums.

    If ``match_schedule`` is provided (list of {stadium_id, date}), each
    stadium's forecast is fetched for its scheduled match date; otherwise
    we use today + 3 days as a default lookahead.
    """
    stadiums = _load_stadiums()

    # Build a stadium-id → match-date lookup from the existing predictions
    if match_schedule is None:
        try:
            with open(DATA_DIR / "predictions.json", encoding="utf-8") as f:
                match_schedule = json.load(f)
        except Exception:
            match_schedule = []

    sid_to_date: dict[str, str] = {}
    for m in match_schedule:
        sid = m.get("stadium_id") or m.get("stadium")
        if sid:
            # Map stadium name → id in stadium_meta
            for s in stadiums:
                if s["name"] == m.get("stadium") or s["city"] == m.get("stadium_city"):
                    sid_to_date[s["id"]] = m["date"]
                    break
            else:
                if sid in {s["id"] for s in stadiums}:
                    sid_to_date.setdefault(sid, m["date"])

    # Fallback: next 3 days for any stadium without an assigned match date
    default_date = (date.today() + timedelta(days=3)).isoformat()

    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for stadium in stadiums:
            sid = stadium["id"]
            target = sid_to_date.get(sid, default_date)
            lat = stadium["coordinates"]["lat"]
            lng = stadium["coordinates"]["lng"]

            logger.info("Fetching weather for %s (%s) on %s", stadium["name"], sid, target)
            w = await fetch_weather_for_stadium(client, lat, lng, target)

            entry = {
                "stadium_id": sid,
                "stadium_name": stadium["name"],
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "target_date": target,
                "weather": w,
            }
            results.append(entry)

    # Persist to cache
    cache_path = DATA_DIR / "weather_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "stadiums": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Weather cache written to %s (%d stadiums)", cache_path, len(results))
    return results


def load_weather_cache() -> dict | None:
    """Return the cached weather snapshot, or None if unavailable."""
    cache_path = DATA_DIR / "weather_cache.json"
    if not cache_path.exists():
        return None
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def get_weather_for_stadium(stadium_id: str) -> dict | None:
    """Convenience: return cached weather for a single stadium."""
    cache = load_weather_cache()
    if cache is None:
        return None
    for s in cache.get("stadiums", []):
        if s["stadium_id"] == stadium_id:
            return s["weather"]
    return None
