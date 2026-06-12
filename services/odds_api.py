"""
The Odds API client — v1.0.0.

Wraps the-odds-api.com v4 endpoints for live match odds, outrights,
and scores. Used to cross-reference model predictions with market data.

Endpoints used:
  GET /v4/sports                                     — list available sports
  GET /v4/sports/{sport_key}/odds                    — match odds (h2h/spreads/totals)
  GET /v4/sports/{sport_key}/scores                  — scores/results
  GET /v4/sports/soccer_fifa_world_cup_winner/odds   — outright winner futures

Usage:
    client = OddsAPIClient(api_key="...")
    matches = await client.get_match_odds()
    outrights = await client.get_outrights()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

_logger = logging.getLogger("odds_api")

BASE_URL = "https://api.the-odds-api.com/v4"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Sport keys
WC_MATCHES = "soccer_fifa_world_cup"
WC_WINNER = "soccer_fifa_world_cup_winner"
WC_SCORES = "soccer_fifa_world_cup"

# Team name mapping: odds API name → internal name
TEAM_NAME_MAP: dict[str, str] = {
    "USA": "United States",
    "USMNT": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
    "Cape Verde": "Cape Verde",
}


class OddsAPIError(Exception):
    """Raised when the Odds API returns an error."""


class OddsAPIClient:
    """Async client for The Odds API v4."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_timeout: int = 15,
        max_retries: int = 2,
    ):
        self._api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self._timeout = request_timeout
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_match_odds(
        self,
        markets: tuple[str, ...] = ("h2h", "spreads", "totals"),
        regions: tuple[str, ...] = ("us", "uk"),
    ) -> list[dict]:
        """Fetch current match odds for all World Cup 2026 matches."""
        markets_str = ",".join(markets)
        regions_str = ",".join(regions)
        path = f"/sports/{WC_MATCHES}/odds"
        params = {
            "regions": regions_str,
            "markets": markets_str,
            "oddsFormat": "decimal",
        }
        return await self._get(path, params)

    async def get_outrights(
        self,
        regions: tuple[str, ...] = ("us", "uk"),
    ) -> list[dict]:
        """Fetch World Cup outright winner futures."""
        regions_str = ",".join(regions)
        path = f"/sports/{WC_WINNER}/odds"
        params = {
            "regions": regions_str,
            "markets": "outrights",
            "oddsFormat": "decimal",
        }
        return await self._get(path, params)

    async def get_scores(self) -> list[dict]:
        """Fetch live scores/results for World Cup matches."""
        path = f"/sports/{WC_SCORES}/scores"
        return await self._get(path, {"daysFrom": "3"})

    # ------------------------------------------------------------------
    # Transformed data for internal use
    # ------------------------------------------------------------------

    async def get_best_odds_summary(self) -> dict[str, dict]:
        """
        Return a lookup of match_key → best odds summary.

        For each match, extracts the best (lowest margin) h2h prices
        across all bookmakers and computes market-implied probabilities.
        """
        try:
            raw = await self.get_match_odds()
        except (OddsAPIError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            _logger.warning("Failed to fetch match odds: %s", e)
            return {}

        summary: dict[str, dict] = {}
        for game in raw:
            match_key = _build_match_key(game)
            if not match_key:
                continue

            h2h_best = _best_h2h(game)
            spreads_best = _best_spreads(game)
            totals_best = _best_totals(game)

            implied_probs = _implied_probabilities(h2h_best) if h2h_best else None

            summary[match_key] = {
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "commence_time": game.get("commence_time"),
                "bookmaker_count": len(game.get("bookmakers", [])),
                "h2h": h2h_best,
                "spreads": spreads_best,
                "totals": totals_best,
                "implied_probs": implied_probs,
            }
        return summary

    async def get_outright_summary(self) -> dict:
        """Return World Cup winner odds averaged across bookmakers."""
        try:
            raw = await self.get_outrights()
        except (OddsAPIError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            _logger.warning("Failed to fetch outrights: %s", e)
            return {"teams": {}, "top_5": [], "updated_at": None}

        if not raw:
            return {"teams": {}, "top_5": [], "updated_at": None}

        game = raw[0]
        team_prices: dict[str, list[float]] = {}

        for bk in game.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    name = _map_team_name(outcome["name"])
                    if name not in team_prices:
                        team_prices[name] = []
                    team_prices[name].append(outcome["price"])

        teams = {}
        for name, prices in team_prices.items():
            avg_price = sum(prices) / len(prices)
            best_price = min(prices)
            implied_prob = round(1.0 / avg_price * 100, 1)
            teams[name] = {
                "avg_price": round(avg_price, 2),
                "best_price": best_price,
                "implied_prob_pct": implied_prob,
                "bookmaker_count": len(prices),
            }

        top_5 = sorted(teams.items(), key=lambda x: x[1]["avg_price"])[:5]
        top_5_list = [{"team": t, **d} for t, d in top_5]

        return {
            "teams": teams,
            "top_5": top_5_list,
            "updated_at": game.get("bookmakers", [{}])[0].get("last_update") if game.get("bookmakers") else None,
        }

    async def get_scores_summary(self) -> list[dict]:
        """Return completed match scores from the API."""
        try:
            raw = await self.get_scores()
        except (OddsAPIError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            _logger.warning("Failed to fetch scores: %s", e)
            return []

        results: list[dict] = []
        for game in raw:
            if not game.get("completed"):
                continue
            results.append({
                "home_team": _map_team_name(game.get("home_team", "")),
                "away_team": _map_team_name(game.get("away_team", "")),
                "home_score": game.get("scores", {}).get("home_score"),
                "away_score": game.get("scores", {}).get("away_score"),
                "completed": game.get("completed"),
                "commence_time": game.get("commence_time"),
            })
        return results

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict | None = None) -> list[dict]:
        """Authenticated GET to The Odds API."""
        if not self._api_key:
            raise OddsAPIError("No API key configured. Set ODDS_API_KEY env var.")

        if params is None:
            params = {}
        params["apiKey"] = self._api_key

        url = f"{BASE_URL}{path}"

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as session:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 401:
                            raise OddsAPIError("Invalid API key")
                        if resp.status == 422:
                            body = await resp.text()
                            raise OddsAPIError(f"Invalid parameters: {body[:300]}")
                        if resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", "60"))
                            _logger.warning("Odds API rate limit; waiting %ds", retry_after)
                            await asyncio.sleep(retry_after)
                            continue
                        if resp.status != 200:
                            body = await resp.text()
                            _logger.warning("Odds API HTTP %d for %s: %s", resp.status, path, body[:200])
                            return []
                        return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)

        _logger.warning("Odds API unreachable after %d attempts: %s", self._max_retries + 1, last_err)
        raise OddsAPIError(f"Unreachable: {last_err}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_match_key(game: dict) -> str | None:
    """Build internal match key from odds API game data."""
    home = _map_team_name(game.get("home_team", ""))
    away = _map_team_name(game.get("away_team", ""))
    commence = game.get("commence_time", "")
    if not home or not away or not commence:
        return None
    try:
        date_str = commence[:10]
    except (ValueError, IndexError):
        return None
    return f"{date_str}_{home}_{away}"


def _map_team_name(raw: str) -> str:
    """Normalize team name from odds API to internal format."""
    if not raw:
        return raw
    return TEAM_NAME_MAP.get(raw, raw)


def _best_h2h(game: dict) -> dict | None:
    """Extract best h2h prices (most favorable for bettor) across bookmakers."""
    home_prices: list[float] = []
    away_prices: list[float] = []
    draw_prices: list[float] = []

    for bk in game.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome["name"]
                price = outcome["price"]
                # Match outcome to home/draw/away
                if _team_fuzzy(name, game["home_team"]):
                    home_prices.append(price)
                elif _team_fuzzy(name, game["away_team"]):
                    away_prices.append(price)
                elif name.lower() == "draw":
                    draw_prices.append(price)

    if not home_prices or not away_prices or not draw_prices:
        return None

    return {
        "home": round(max(home_prices), 2),  # best = highest decimal odds
        "away": round(max(away_prices), 2),
        "draw": round(max(draw_prices), 2),
    }


def _best_spreads(game: dict) -> dict | None:
    """Extract best spread (Asian handicap) prices."""
    home_prices: list[float] = []
    away_prices: list[float] = []
    home_points: list[float] = []
    away_points: list[float] = []

    for bk in game.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "spreads":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome["name"]
                price = outcome["price"]
                point = outcome.get("point", 0)
                if _team_fuzzy(name, game["home_team"]):
                    home_prices.append(price)
                    home_points.append(point)
                elif _team_fuzzy(name, game["away_team"]):
                    away_prices.append(price)
                    away_points.append(point)

    if not home_prices or not away_prices:
        return None

    # Find the most common spread line
    return {
        "home_price": round(max(home_prices), 2),
        "away_price": round(max(away_prices), 2),
        "home_point": home_points[0] if home_points else 0,
        "away_point": away_points[0] if away_points else 0,
    }


def _best_totals(game: dict) -> dict | None:
    """Extract best over/under prices."""
    over_prices: list[float] = []
    under_prices: list[float] = []
    points: list[float] = []

    for bk in game.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome["name"]
                price = outcome["price"]
                point = outcome.get("point", 2.5)
                if name == "Over":
                    over_prices.append(price)
                    points.append(point)
                elif name == "Under":
                    under_prices.append(price)

    if not over_prices or not under_prices:
        return None

    return {
        "over": round(max(over_prices), 2),
        "under": round(max(under_prices), 2),
        "point": points[0] if points else 2.5,
    }


def _implied_probabilities(h2h: dict) -> dict | None:
    """
    Convert best h2h decimal odds to market-implied probabilities.
    Removes overround (vigorish) via proportional scaling.
    """
    if not h2h:
        return None

    raw_home = 1.0 / h2h["home"]
    raw_away = 1.0 / h2h["away"]
    raw_draw = 1.0 / h2h["draw"]

    total = raw_home + raw_away + raw_draw
    if total == 0:
        return None

    return {
        "home": round(raw_home / total * 100, 1),
        "away": round(raw_away / total * 100, 1),
        "draw": round(raw_draw / total * 100, 1),
    }


def _team_fuzzy(a: str, b: str) -> bool:
    """Case-insensitive team name match."""
    import re
    a_norm = re.sub(r"[^a-z]", "", a.lower())
    b_norm = re.sub(r"[^a-z]", "", b.lower())
    return a_norm == b_norm or a_norm in b_norm or b_norm in a_norm


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: OddsAPIClient | None = None


def get_odds_client() -> OddsAPIClient:
    global _client
    if _client is None:
        api_key = os.getenv("ODDS_API_KEY", "33244de2bcd351e42b4a202c211c55ed")
        _client = OddsAPIClient(api_key=api_key)
    return _client
