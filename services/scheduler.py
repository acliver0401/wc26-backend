"""
APScheduler-powered daily pipeline trigger + high-frequency lineup poller.

Runs ``execute_daily_pipeline()`` on a cron schedule (default: 05:00
Asia/Shanghai = 21:00 UTC), which:
  1. Backtests yesterday's completed matches
  2. Refreshes live weather + injury snapshots
  3. Polls for official lineups (T-75min window)
  4. Re-predicts all scheduled matches with fresh data
  5. Writes ``data/latest_predictions.json`` + ``backtest_history.json``

Also manages a high-frequency lineup poller (every 5 min) that activates
75 minutes before each match kickoff.
"""

from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pytz import timezone as tz

from pipeline.cron_update import execute_daily_pipeline
from pipeline.lineup_fetcher import run_lineup_poll

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None

# UTC cron for 05:00 Asia/Shanghai (= 21:00 UTC)
DEFAULT_CRON = "0 21 * * *"
DEFAULT_TZ = "Asia/Shanghai"

# Lineup polling interval (seconds)
LINEUP_POLL_INTERVAL = 300  # 5 minutes


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
    """Start the background scheduler with Asia/Shanghai timezone."""
    global _scheduler

    if _scheduler is not None:
        logger.warning("Scheduler already running; skipping duplicate start.")
        return _scheduler

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _run_async_refresh,
        trigger=CronTrigger.from_crontab(cron_expr, timezone=tz(DEFAULT_TZ)),
        id="wc26_daily_pipeline",
        name="WC26 daily pipeline",
        replace_existing=True,
        misfire_grace_time=600,
    )
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


def start_lineup_poller(interval_seconds: int = LINEUP_POLL_INTERVAL) -> None:
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
# Run-once helper (called at startup so there is data immediately)
# ---------------------------------------------------------------------------

async def run_initial_refresh() -> dict:
    """Run the full pipeline once at application startup."""
    logger.info("Running initial pipeline on startup...")
    return await run_pipeline()
