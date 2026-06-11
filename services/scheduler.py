"""
APScheduler-powered data refresh coordinator.

Runs ``fetch_and_update_all_data()`` on a cron schedule, which:
  1. Fetches live weather from Open-Meteo for all 16 stadiums
  2. Refreshes mock injury / team-status snapshots
  3. Re-runs the ML prediction pipeline for all scheduled matches
  4. Writes ``data/latest_predictions.json`` for the API to serve

Also exposes ``start_scheduler()`` / ``shutdown_scheduler()`` for the
FastAPI lifespan, and a helper to build the match schedule from the
reference predictions file (stadium_id ← stadium assignment).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from models.predictor import predict_match
from services.weather import fetch_all_weather, load_weather_cache, get_weather_for_stadium
from services.injuries import generate_injuries, load_injury_cache, get_injuries_for_team

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_scheduler: BackgroundScheduler | None = None


# ---------------------------------------------------------------------------
# Match schedule builder
# ---------------------------------------------------------------------------

def build_match_schedule() -> list[dict]:
    """
    Derive the match schedule from the reference predictions file.

    Each entry has: {date, home, away, stadium_id}
    """
    ref_path = DATA_DIR / "predictions.json"
    if not ref_path.exists():
        logger.warning("No predictions.json found; cannot build schedule.")
        return []

    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)

    stadiums = _load_stadiums_by_name()
    schedule: list[dict] = []
    for m in ref:
        sid = m.get("stadium_id") or _resolve_stadium_id(m.get("stadium", ""), stadiums)
        schedule.append({
            "date": m["date"],
            "home": m["home"],
            "away": m["away"],
            "stadium_id": sid,
        })
    return schedule


def _load_stadiums_by_name() -> dict[str, str]:
    """Map stadium name → id."""
    with open(DATA_DIR / "stadium_meta.json", encoding="utf-8") as f:
        all_s = json.load(f)
    return {s["name"]: s["id"] for s in all_s}


def _resolve_stadium_id(name_or_id: str, lookup: dict[str, str]) -> str:
    # If it's already a valid stadium ID, return it directly
    valid_ids = set(lookup.values())
    if name_or_id in valid_ids:
        return name_or_id
    # Try exact name match
    if name_or_id in lookup:
        return lookup[name_or_id]
    # Fuzzy match
    for k, v in lookup.items():
        if name_or_id in k or k in name_or_id:
            return v
    # Try matching against city
    all_stadiums = _load_stadiums_list()
    for s in all_stadiums:
        if s["city"] == name_or_id or s["name"] == name_or_id:
            return s["id"]
    return "att"  # fallback


def _load_stadiums_list() -> list[dict]:
    with open(DATA_DIR / "stadium_meta.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core update function
# ---------------------------------------------------------------------------

async def fetch_and_update_all_data() -> dict:
    """
    Run the full data refresh pipeline.

    Returns a summary dict with counts / timestamps for logging.
    """
    started = datetime.utcnow()
    logger.info("=== Data refresh started at %s ===", started.isoformat())

    summary: dict = {
        "started_at": started.isoformat() + "Z",
        "weather": {"success": 0, "failed": 0},
        "injuries": {"teams": 0},
        "predictions": {"total": 0, "regenerated": 0},
    }

    # 1. Fetch live weather ---------------------------------------------------
    try:
        schedule = build_match_schedule()
        weather_results = await fetch_all_weather(match_schedule=schedule)
        for w in weather_results:
            if w["weather"] is not None:
                summary["weather"]["success"] += 1
            else:
                summary["weather"]["failed"] += 1
        logger.info("Weather: %d ok, %d failed", summary["weather"]["success"], summary["weather"]["failed"])
    except Exception:
        logger.exception("Weather fetch step failed; continuing with stale cache.")

    # 2. Refresh injury / team-status mocks -----------------------------------
    try:
        # Use a seed derived from the current hour so values drift slightly each tick
        hour_seed = hash(datetime.utcnow().strftime("%Y%m%d%H"))
        injuries = generate_injuries(seed=hour_seed)
        summary["injuries"]["teams"] = len(injuries)
        logger.info("Injuries: %d teams updated", summary["injuries"]["teams"])
    except Exception:
        logger.exception("Injury generation failed; continuing with stale cache.")

    # 3. Re-run predictions ---------------------------------------------------
    try:
        schedule = build_match_schedule()
        if not schedule:
            logger.warning("No match schedule available; skipping prediction re-run.")
        else:
            new_predictions = []
            for m in schedule:
                pred = predict_match(
                    home_team=m["home"],
                    away_team=m["away"],
                    match_date=m["date"],
                    stadium_id=m["stadium_id"],
                    # Pass live weather + injuries so they override static defaults
                    weather_override=get_weather_for_stadium(m["stadium_id"]),
                    injury_override={
                        m["home"]: get_injuries_for_team(m["home"]),
                        m["away"]: get_injuries_for_team(m["away"]),
                    },
                )
                new_predictions.append(pred)

            # Write cache
            out_path = DATA_DIR / "latest_predictions.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "updated_at": datetime.utcnow().isoformat() + "Z",
                        "count": len(new_predictions),
                        "predictions": new_predictions,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            summary["predictions"]["total"] = len(new_predictions)
            summary["predictions"]["regenerated"] = len(new_predictions)
            logger.info("Predictions: %d matches written to %s", len(new_predictions), out_path)
    except Exception:
        logger.exception("Prediction re-run failed.")

    completed = datetime.utcnow()
    summary["completed_at"] = completed.isoformat() + "Z"
    summary["duration_seconds"] = (completed - started).total_seconds()
    logger.info("=== Data refresh completed in %.1fs ===", summary["duration_seconds"])
    return summary


# ---------------------------------------------------------------------------
# Scheduler lifecycle (called from FastAPI lifespan)
# ---------------------------------------------------------------------------

def start_scheduler(cron_expr: str = "0 */6 * * *") -> BackgroundScheduler:
    """
    Start the background scheduler.

    Default: every 6 hours.  Pass ``"0 2 * * *"`` for daily at 02:00.
    """
    global _scheduler

    if _scheduler is not None:
        logger.warning("Scheduler already running; skipping duplicate start.")
        return _scheduler

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _run_async_refresh,
        trigger=CronTrigger.from_crontab(cron_expr),
        id="wc26_data_refresh",
        name="WC26 data refresh",
        replace_existing=True,
        misfire_grace_time=300,  # 5 min grace
    )
    _scheduler.start()
    logger.info("Scheduler started (cron: %s)", cron_expr)
    return _scheduler


def shutdown_scheduler() -> None:
    """Gracefully stop the background scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler shut down.")


def _run_async_refresh() -> None:
    """Wrapper so APScheduler can call the async refresh function."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(fetch_and_update_all_data())


# ---------------------------------------------------------------------------
# Run-once helper (called at startup so there is data immediately)
# ---------------------------------------------------------------------------

async def run_initial_refresh() -> dict:
    """Run the full pipeline once at application startup."""
    logger.info("Running initial data refresh on startup...")
    return await fetch_and_update_all_data()
