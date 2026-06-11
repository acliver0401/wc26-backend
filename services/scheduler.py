"""
APScheduler-powered daily pipeline trigger.

Runs ``execute_daily_pipeline()`` on a cron schedule (default: 05:00
Asia/Shanghai = 21:00 UTC), which:
  1. Backtests yesterday's completed matches
  2. Refreshes live weather + injury snapshots
  3. Re-predicts all scheduled matches with fresh data
  4. Writes ``data/latest_predictions.json`` + ``backtest_history.json``

Also exposes ``start_scheduler()`` / ``shutdown_scheduler()`` for the
FastAPI lifespan, and ``run_initial_refresh()`` for startup.
"""

from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone as tz

from pipeline.cron_update import execute_daily_pipeline

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None

# UTC cron for 05:00 Asia/Shanghai (= 21:00 UTC)
DEFAULT_CRON = "0 21 * * *"
DEFAULT_TZ = "Asia/Shanghai"


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
# Run-once helper (called at startup so there is data immediately)
# ---------------------------------------------------------------------------

async def run_initial_refresh() -> dict:
    """Run the full pipeline once at application startup."""
    logger.info("Running initial pipeline on startup...")
    return await run_pipeline()
