"""
APScheduler-powered daily pipeline trigger + match-day live refresh.

Runs ``execute_daily_pipeline()`` on a cron schedule (default: 05:00
Asia/Shanghai = 21:00 UTC), which:
  1. Backtests yesterday's completed matches
  2. Refreshes live weather + injury snapshots
  3. Polls for official lineups (T-75min window)
  4. Re-predicts all scheduled matches with fresh data
  5. Writes ``data/latest_predictions.json`` + ``backtest_history.json``

During tournament match days (June 11 – July 19, 2026), a lightweight
live-refresh runs every 5 minutes to fetch real-time results + lineups
without re-running the heavy weather/injury/prediction pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pytz import timezone as tz

from pipeline.cron_update import execute_daily_pipeline
from pipeline.lineup_fetcher import run_lineup_poll
from pipeline.result_fetcher import fetch_live_results

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# UTC cron for 05:00 Asia/Shanghai (= 21:00 UTC)
DEFAULT_CRON = "0 21 * * *"
DEFAULT_TZ = "Asia/Shanghai"

# Live refresh interval (seconds) — runs during tournament match days
LIVE_REFRESH_INTERVAL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Core refresh — delegates to the daily pipeline
# ---------------------------------------------------------------------------

async def run_pipeline() -> dict:
    """Thin wrapper so the scheduler and startup can share one call site."""
    return await execute_daily_pipeline()


# ---------------------------------------------------------------------------
# Scheduler lifecycle (called from FastAPI lifespan)
# ---------------------------------------------------------------------------

def start_scheduler(cron_expr: str = DEFAULT_CRON) -> BackgroundScheduler:
    """Start the background scheduler with Asia/Shanghai timezone,
    plus a match-day live-refresh every 5 minutes."""
    global _scheduler

    if _scheduler is not None:
        logger.warning("Scheduler already running; skipping duplicate start.")
        return _scheduler

    _scheduler = BackgroundScheduler(daemon=True)

    # Daily full pipeline at 05:00 CST
    _scheduler.add_job(
        _run_async_refresh,
        trigger=CronTrigger.from_crontab(cron_expr, timezone=tz(DEFAULT_TZ)),
        id="wc26_daily_pipeline",
        name="WC26 daily pipeline",
        replace_existing=True,
        misfire_grace_time=600,
    )

    # Match-day live refresh every 5 minutes (results + lineups only)
    if _is_tournament_active():
        _scheduler.add_job(
            _run_async_live_refresh,
            trigger=IntervalTrigger(seconds=LIVE_REFRESH_INTERVAL),
            id="wc26_live_refresh",
            name="WC26 live refresh (results + lineups)",
            replace_existing=True,
            misfire_grace_time=120,
        )
        logger.info("Live-refresh enabled (tournament is active)")

    _scheduler.start()
    logger.info("Scheduler started (cron: %s, tz: %s)", cron_expr, DEFAULT_TZ)
    return _scheduler


def shutdown_scheduler() -> None:
    """Gracefully stop the background scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler shut down.")


def _run_async_refresh() -> None:
    """Wrapper so APScheduler can call the async pipeline."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(run_pipeline())


# ---------------------------------------------------------------------------
# High-frequency lineup poller (every 5 min, active T-75min to kickoff)
# ---------------------------------------------------------------------------

_lineup_job_id = "wc26_lineup_poll"


def start_lineup_poller(interval_seconds: int = LIVE_REFRESH_INTERVAL) -> None:
    """Start the high-frequency lineup polling job."""
    global _scheduler
    if _scheduler is None:
        logger.warning("Scheduler not running; cannot start lineup poller.")
        return

    existing = _scheduler.get_job(_lineup_job_id)
    if existing is not None:
        logger.info("Lineup poller already running.")
        return

    _scheduler.add_job(
        _run_async_lineup_poll,
        trigger=IntervalTrigger(seconds=interval_seconds),
        id=_lineup_job_id,
        name="WC26 lineup polling (every 5 min)",
        replace_existing=True,
        misfire_grace_time=120,
    )
    logger.info("Lineup poller started (interval: %ds)", interval_seconds)


def shutdown_lineup_poller() -> None:
    """Stop the high-frequency lineup polling job."""
    global _scheduler
    if _scheduler is None:
        return
    job = _scheduler.get_job(_lineup_job_id)
    if job is not None:
        _scheduler.remove_job(_lineup_job_id)
        logger.info("Lineup poller stopped.")


def _run_async_lineup_poll() -> None:
    """Wrapper so APScheduler can call the async lineup poll."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(run_lineup_poll())


# ---------------------------------------------------------------------------
# Match-day live refresh — results + lineups every 5 min
# ---------------------------------------------------------------------------

def _is_tournament_active() -> bool:
    """Return True if we're within the WC 2026 tournament window."""
    now = datetime.now(timezone.utc)
    start = datetime(2026, 6, 10, tzinfo=timezone.utc)  # day before first match
    end = datetime(2026, 7, 20, tzinfo=timezone.utc)    # day after final
    return start <= now <= end


async def _run_live_refresh_pipeline() -> dict:
    """Lightweight refresh: results + lineups only, no heavy recomputation."""
    logger.debug("Running live refresh...")

    result_summary = {"results": 0, "lineups": 0, "modified": 0}

    try:
        # 1. Fetch live results
        live_results = await fetch_live_results()
        results_lookup: dict[str, dict] = {}
        for r in live_results:
            key = f"{r['date']}_{r['home']}_{r['away']}"
            if r.get("status") and r["status"] != "NS":
                results_lookup[key] = r

        # 2. Poll lineups
        lineup_summary = await run_lineup_poll()

        # 3. Merge into latest_predictions.json
        predictions_path = DATA_DIR / "latest_predictions.json"
        if predictions_path.exists():
            with open(predictions_path, encoding="utf-8") as f:
                cache = json.load(f)

            predictions = cache.get("predictions", [])
            modified = 0
            for p in predictions:
                result_key = f"{p['date']}_{p['home']}_{p['away']}"
                if result_key in results_lookup:
                    r = results_lookup[result_key]
                    if not p.get("result") or p["result"].get("status") != r.get("status"):
                        p["result"] = {
                            "home_score": r.get("home_score"),
                            "away_score": r.get("away_score"),
                            "outcome": r.get("outcome"),
                            "status": r.get("status", "FT"),
                        }
                        modified += 1

            if modified > 0 or lineup_summary.get("lineups_fetched", 0) > 0:
                live_count = sum(1 for p in predictions if p.get("prediction_status") == "Live-Lineup")
                cache["updated_at"] = datetime.now(timezone.utc).isoformat() + "Z"
                cache["live_lineup_count"] = live_count
                with open(predictions_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)

            result_summary = {
                "results": len(results_lookup),
                "lineups": lineup_summary.get("lineups_fetched", 0),
                "modified": modified,
            }
    except Exception:
        logger.exception("Live refresh failed")

    return result_summary


def _run_async_live_refresh() -> None:
    """Wrapper for the live refresh pipeline."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(_run_live_refresh_pipeline())


# ---------------------------------------------------------------------------
# Run-once helper (called at startup so there is data immediately)
# ---------------------------------------------------------------------------

async def run_initial_refresh() -> dict:
    """Run the full pipeline once at application startup."""
    logger.info("Running initial pipeline on startup...")
    return await run_pipeline()
