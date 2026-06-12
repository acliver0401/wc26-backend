"""
Daily pipeline orchestrator — coordinates backtest, data refresh,
model retraining, and cache regeneration in a single workflow.

Usage::

    await execute_daily_pipeline()

Called by:
  1. APScheduler (daily at 05:00 Asia/Shanghai = 21:00 UTC)
  2. POST /api/admin/force-refresh  (on-demand trigger, e.g. cron-job.org)
"""

from __future__ import annotations

import json
import logging
from datetime import date as date_type, datetime, timedelta
from pathlib import Path
from typing import Optional

from models.predictor import predict_match
from pipeline.backtest import run_daily_backtest, get_backtest_summary
from pipeline.result_fetcher import fetch_yesterday_results, fetch_live_results
from pipeline.lineup_fetcher import run_lineup_poll, get_lineup_multipliers
from services.weather import fetch_all_weather, load_weather_cache, get_weather_for_stadium
from services.injuries import generate_injuries, load_injury_cache, get_injuries_for_team

_logger = logging.getLogger("pipeline")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def execute_daily_pipeline(
    reference_date: Optional[date_type] = None,
    *,
    skip_backtest: bool = False,
    skip_refresh: bool = False,
    skip_predict: bool = False,
) -> dict:
    """
    Execute the complete daily pipeline.

    Parameters
    ----------
    reference_date : date | None
        The "today" for which to run.  Defaults to ``date.today()``.
    skip_backtest / skip_refresh / skip_predict : bool
        Allow skipping phases for faster testing.

    Returns
    -------
    dict
        Summary with status of each phase.
    """
    today = reference_date or date_type.today()
    started = datetime.utcnow()

    _logger.info("=" * 64)
    _logger.info("DAILY PIPELINE START — %s", today.isoformat())
    _logger.info("=" * 64)

    summary: dict = {
        "pipeline_started": started.isoformat() + "Z",
        "reference_date": today.isoformat(),
        "phases": {},
    }

    # =================================================================
    # Phase 1 — Backtest yesterday's completed matches
    # =================================================================
    if not skip_backtest:
        try:
            bt = run_daily_backtest(reference_date=today)
            summary["phases"]["backtest"] = bt
            _logger.info(
                "Phase 1 BACKTEST: %s — %d matches evaluated",
                bt.get("status"), bt.get("matches_evaluated", 0),
            )
        except Exception:
            _logger.exception("Phase 1 BACKTEST failed")
            summary["phases"]["backtest"] = {"status": "error"}

    # =================================================================
    # Phase 1b — Fetch real match results from Flashscore
    # =================================================================
    results_lookup: dict[str, dict] = {}
    try:
        live_results = await fetch_live_results()
        for r in live_results:
            key = f"{r['date']}_{r['home']}_{r['away']}"
            if r.get("status") and r["status"] != "NS":
                results_lookup[key] = r
        summary["phases"]["results"] = {
            "fetched": len(live_results),
            "completed": len(results_lookup),
        }
        _logger.info(
            "Phase 1b RESULTS: %d fetched, %d completed",
            len(live_results), len(results_lookup),
        )
    except Exception:
        _logger.exception("Phase 1b RESULTS fetch failed")
        summary["phases"]["results"] = {"status": "error"}

    # =================================================================
    # Phase 2 — Refresh live data for upcoming matches
    # =================================================================
    if not skip_refresh:
        try:
            # 2a. Weather
            schedule = _build_schedule()
            weather_results = await fetch_all_weather(match_schedule=schedule)
            w_ok = sum(1 for w in weather_results if w["weather"] is not None)
            summary["phases"]["weather"] = {"ok": w_ok, "total": len(weather_results)}

            # 2b. Injuries
            hour_seed = hash(datetime.utcnow().strftime("%Y%m%d%H"))
            injuries = generate_injuries(seed=hour_seed)
            summary["phases"]["injuries"] = {"teams": len(injuries)}

            # 2c. Lineup polling
            try:
                lineup_summary = await run_lineup_poll()
                summary["phases"]["lineups"] = lineup_summary
                _logger.info(
                    "Phase 2c LINEUPS: %d checked, %d fetched",
                    lineup_summary.get("matches_checked", 0),
                    lineup_summary.get("lineups_fetched", 0),
                )
            except Exception:
                _logger.exception("Phase 2c LINEUP poll failed")
                summary["phases"]["lineups"] = {"status": "error"}

            # 2d. Market odds (the-odds-api.com)
            try:
                from services.odds_api import get_odds_client
                client = get_odds_client()
                odds_data = await client.get_best_odds_summary()
                outright_data = await client.get_outright_summary()
                odds_cache = {
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                    "matches": odds_data,
                    "outrights": outright_data,
                }
                cache_path = DATA_DIR / "odds_cache.json"
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(odds_cache, f, ensure_ascii=False, indent=2)
                summary["phases"]["odds"] = {
                    "matches_cached": len(odds_data),
                    "outrights_teams": len(outright_data.get("teams", {})),
                }
                _logger.info(
                    "Phase 2d ODDS: %d matches, %d outrights teams",
                    len(odds_data), len(outright_data.get("teams", {})),
                )
            except Exception:
                _logger.exception("Phase 2d ODDS fetch failed")
                summary["phases"]["odds"] = {"status": "error"}

            _logger.info(
                "Phase 2 REFRESH: weather %d/%d, injuries %d teams, odds %d matches",
                w_ok, len(weather_results), len(injuries),
                summary["phases"].get("odds", {}).get("matches_cached", 0),
            )
        except Exception:
            _logger.exception("Phase 2 REFRESH failed")
            summary["phases"]["weather"] = {"status": "error"}
            summary["phases"]["injuries"] = {"status": "error"}
            summary["phases"]["lineups"] = {"status": "error"}

    # =================================================================
    # Phase 3 — Re-predict all future matches (Poisson model)
    # =================================================================
    if not skip_predict:
        try:
            schedule = _build_schedule()
            if not schedule:
                _logger.warning("No schedule; skipping prediction phase.")
                summary["phases"]["predictions"] = {"status": "no_schedule"}
            else:
                new_predictions = []
                for m in schedule:
                    pred = predict_match(
                        home_team=m["home"],
                        away_team=m["away"],
                        match_date=m["date"],
                        stadium_id=m["stadium_id"],
                        weather_override=get_weather_for_stadium(m["stadium_id"]),
                        injury_override={
                            m["home"]: get_injuries_for_team(m["home"]),
                            m["away"]: get_injuries_for_team(m["away"]),
                        },
                    )
                    # Merge real result if available
                    result_key = f"{m['date']}_{m['home']}_{m['away']}"
                    if result_key in results_lookup:
                        r = results_lookup[result_key]
                        pred["result"] = {
                            "home_score": r.get("home_score"),
                            "away_score": r.get("away_score"),
                            "outcome": r.get("outcome"),
                            "status": r.get("status", "FT"),
                        }
                    new_predictions.append(pred)

                # Attach backtest metadata
                # Count Live-Lineup matches
                live_count = sum(1 for p in new_predictions if p.get("prediction_status") == "Live-Lineup")

                bt_summary = get_backtest_summary()
                out = {
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                    "count": len(new_predictions),
                    "live_lineup_count": live_count,
                    "backtest": {
                        "accuracy": bt_summary["cumulative"]["accuracy"],
                        "total_matches": bt_summary["cumulative"]["total_matches"],
                        "roi": bt_summary["cumulative"]["roi"],
                    },
                    "predictions": new_predictions,
                }

                out_path = DATA_DIR / "latest_predictions.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

                summary["phases"]["predictions"] = {
                    "total": len(new_predictions),
                    "written": str(out_path),
                }
                _logger.info(
                    "Phase 3 PREDICT: %d matches → %s", len(new_predictions), out_path,
                )
        except Exception:
            _logger.exception("Phase 3 PREDICT failed")
            summary["phases"]["predictions"] = {"status": "error"}

    # =================================================================
    # Final
    # =================================================================
    completed = datetime.utcnow()
    summary["pipeline_completed"] = completed.isoformat() + "Z"
    summary["duration_seconds"] = round((completed - started).total_seconds(), 1)
    _logger.info("DAILY PIPELINE DONE — %.1fs", summary["duration_seconds"])
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_schedule() -> list[dict]:
    """Load the match schedule from static predictions.json."""
    ref_path = DATA_DIR / "predictions.json"
    if not ref_path.exists():
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
    with open(DATA_DIR / "stadium_meta.json", encoding="utf-8") as f:
        all_s = json.load(f)
    return {s["name"]: s["id"] for s in all_s}


def _resolve_stadium_id(name_or_id: str, lookup: dict[str, str]) -> str:
    valid_ids = set(lookup.values())
    if name_or_id in valid_ids:
        return name_or_id
    if name_or_id in lookup:
        return lookup[name_or_id]
    for k, v in lookup.items():
        if name_or_id in k or k in name_or_id:
            return v
    all_stadiums = _load_stadiums_list()
    for s in all_stadiums:
        if s["city"] == name_or_id or s["name"] == name_or_id:
            return s["id"]
    return "att"


def _load_stadiums_list() -> list[dict]:
    with open(DATA_DIR / "stadium_meta.json", encoding="utf-8") as f:
        return json.load(f)
