"""
Backtest engine — evaluates predictions against real match results,
computes rolling accuracy & simulated ROI, and maintains ``backtest_history.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import date as date_type, datetime, timedelta
from pathlib import Path
from typing import Optional

from pipeline.result_fetcher import fetch_yesterday_results

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_daily_backtest(reference_date: Optional[date_type] = None) -> dict:
    """
    Evaluate predictions for matches completed *yesterday* (relative to
    *reference_date*).  Appends a new daily record to ``backtest_history.json``
    and returns a summary dict.
    """
    today = reference_date or date_type.today()
    yesterday = (today - timedelta(days=1)).isoformat()

    # --- 1. Fetch actual results -------------------------------------------
    results = fetch_yesterday_results(reference_date)
    if not results:
        return {"status": "no_matches", "date": yesterday, "evaluated": 0}

    # Build lookup keyed by (home, away, date)
    actual_map: dict[tuple[str, str, str], dict] = {}
    for r in results:
        actual_map[(r["home"], r["away"], r["date"])] = r

    # --- 2. Load predictions that were made --------------------------------
    predictions = _load_latest_predictions()
    pred_map: dict[tuple[str, str, str], dict] = {}
    for p in predictions:
        pred_map[(p["home"], p["away"], p["date"])] = p

    # --- 3. Compare ---------------------------------------------------------
    daily_record = _evaluate(pred_map, actual_map, yesterday)
    if daily_record is None or daily_record.get("matches_evaluated", 0) == 0:
        return {"status": "no_overlap", "date": yesterday, "evaluated": 0}

    # --- 4. Append to history -----------------------------------------------
    history = _load_backtest_history()
    history["daily_records"].append(daily_record)
    _recalc_cumulative(history)
    history["updated_at"] = datetime.utcnow().isoformat() + "Z"

    _save_backtest_history(history)

    _logger.info(
        "Backtest %s: %d matches, accuracy %.1f%%, ROI %+.1f%%",
        yesterday,
        daily_record["matches_evaluated"],
        daily_record["daily_accuracy"] * 100,
        daily_record["daily_roi"] * 100,
    )
    return {"status": "ok", "date": yesterday, **daily_record}


def get_backtest_summary() -> dict:
    """Return cumulative backtest stats for the API."""
    history = _load_backtest_history()
    return {
        "updated_at": history.get("updated_at"),
        "cumulative": history.get("cumulative", _empty_cumulative()),
        "daily_count": len(history.get("daily_records", [])),
    }


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def _evaluate(
    pred_map: dict[tuple[str, str, str], dict],
    actual_map: dict[tuple[str, str, str], dict],
    date_str: str,
) -> Optional[dict]:
    matches: list[dict] = []
    correct_outcomes = 0
    correct_scores = 0

    for key, actual in actual_map.items():
        pred = pred_map.get(key)
        if pred is None:
            continue

        # Outcome comparison
        outcome_ok = pred.get("pred_r") == actual["outcome"]

        # Score comparison
        pred_top_score = _parse_top_score(pred.get("score_probs", {}))
        score_ok = (
            pred_top_score is not None
            and pred_top_score[0] == actual["home_score"]
            and pred_top_score[1] == actual["away_score"]
        )

        if outcome_ok:
            correct_outcomes += 1
        if score_ok:
            correct_scores += 1

        # Simulated ROI (stake 100 units, rough odds from probability)
        stake = 100
        if pred.get("pred_r") == "H":
            imp_odds = 1.0 / max(pred.get("ph", 33) / 100, 0.05)
        elif pred.get("pred_r") == "D":
            imp_odds = 1.0 / max(pred.get("pd", 33) / 100, 0.05)
        else:
            imp_odds = 1.0 / max(pred.get("pa", 33) / 100, 0.05)
        ret = round(stake * imp_odds) if outcome_ok else 0

        matches.append({
            "home": actual["home"],
            "away": actual["away"],
            "predicted_outcome": pred.get("pred_r"),
            "actual_outcome": actual["outcome"],
            "predicted_score": f'{pred_top_score[0]}-{pred_top_score[1]}' if pred_top_score else "?",
            "actual_score": f'{actual["home_score"]}-{actual["away_score"]}',
            "outcome_correct": outcome_ok,
            "score_correct": score_ok,
            "ph": pred.get("ph", 0),
            "pd": pred.get("pd", 0),
            "pa": pred.get("pa", 0),
            "bet_advice": pred.get("bet_advice", ""),
            "stake": stake,
            "return": ret,
            "odds_used": round(imp_odds, 2),
        })

    if not matches:
        return None

    n = len(matches)
    return {
        "date": date_str,
        "matches_evaluated": n,
        "correct_outcomes": correct_outcomes,
        "correct_scores": correct_scores,
        "daily_accuracy": round(correct_outcomes / n, 4),
        "daily_roi": round((sum(m["return"] for m in matches) - n * 100) / (n * 100), 4),
        "matches": matches,
    }


def _parse_top_score(score_probs: dict) -> Optional[tuple[int, int]]:
    if not score_probs:
        return None
    top = max(score_probs.items(), key=lambda x: x[1])[0]
    parts = top.split("-")
    try:
        h = int(parts[0].replace("5+", "5"))
        a = int(parts[1].replace("5+", "5"))
        return h, a
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Backtest-history persistence
# ---------------------------------------------------------------------------


def _empty_cumulative() -> dict:
    return {
        "total_matches": 0,
        "correct_predictions": 0,
        "accuracy": 0.0,
        "total_stake": 0,
        "total_return": 0,
        "roi": 0.0,
    }


def _load_backtest_history() -> dict:
    path = DATA_DIR / "backtest_history.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"updated_at": None, "cumulative": _empty_cumulative(), "daily_records": []}


def _save_backtest_history(history: dict) -> None:
    path = DATA_DIR / "backtest_history.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _recalc_cumulative(history: dict) -> None:
    all_matches: list[dict] = []
    for day in history.get("daily_records", []):
        all_matches.extend(day.get("matches", []))

    n = len(all_matches)
    correct = sum(1 for m in all_matches if m.get("outcome_correct"))
    total_stake = sum(m.get("stake", 100) for m in all_matches)
    total_return = sum(m.get("return", 0) for m in all_matches)

    history["cumulative"] = {
        "total_matches": n,
        "correct_predictions": correct,
        "accuracy": round(correct / n, 4) if n > 0 else 0.0,
        "total_stake": total_stake,
        "total_return": total_return,
        "roi": round((total_return - total_stake) / total_stake, 4) if total_stake > 0 else 0.0,
    }


def _load_latest_predictions() -> list[dict]:
    """Read predictions from the scheduler cache (or static fallback)."""
    cache = DATA_DIR / "latest_predictions.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("predictions", [])

    static = DATA_DIR / "predictions.json"
    if static.exists():
        with open(static, encoding="utf-8") as f:
            return json.load(f)
    return []
