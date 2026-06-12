"""
Multi-source lineup client — v5.0.0.

Priority (updated 2026-06-12):
  1. Flashscore.com internal JSON API  (free, no key, real-time confirmed XIs)
  2. Zafronix World Cup API             (free tier, 1000 req/day, roster data)
  3. Local player_db.json simulation     (offline fallback)

Removed: api-football.com (free tier too limited at 100 calls/day),
         Dongqiudi (reverse-engineered API — broke, auth changed).

Usage:
    client = LineupAPIClient()
    lineup = await client.fetch_lineup(home_team="Mexico", away_team="South Africa",
                                        match_date="2026-06-11")
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

_logger = logging.getLogger("lineup_api")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class LineupAPIError(Exception):
    """Raised when all lineup sources fail."""


class LineupNotAvailable(Exception):
    """Raised when lineups haven't been published yet."""


class LineupAPIClient:
    """
    Multi-source lineup client with automatic fallback.

    v5.0.0 priority:
      1. Flashscore internal API  — confirmed XIs, free, no key
      2. Zafronix API             — roster-based simulation, free 1K/day
      3. player_db simulation     — offline deterministic fallback
    """

    def __init__(
        self,
        zafronix_api_key: str | None = None,
        *,
        request_timeout: int = 15,
        max_retries: int = 2,
    ):
        self._zf_key = zafronix_api_key
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._flashscore: object | None = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_lineup(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        *,
        stadium_id: str = "",
    ) -> dict | None:
        """
        Fetch official starting lineups for a match.

        Returns a dict with:
          home_lineup:  {formation, players: [{name, pos, number, rating}]}
          away_lineup:  {formation, players: [{name, pos, number, rating}]}
          status:       "Live-Lineup"
          fetched_at:   ISO timestamp
          source:       "flashscore" | "zafronix" | "player_db"

        Returns None if lineups are not yet available.
        """
        match_key = f"{match_date}_{home_team}_{away_team}"

        # ── 1. Flashscore (free, real-time confirmed XIs) ──────────────
        try:
            result = await self._fetch_flashscore(home_team, away_team, match_date)
            if result:
                _logger.info("Lineup from Flashscore for %s", match_key)
                return result
        except LineupNotAvailable:
            _logger.info("Flashscore: lineups not yet available for %s", match_key)
        except Exception as e:
            _logger.warning("Flashscore failed for %s: %s", match_key, e)

        # ── 2. Zafronix (free tier, roster-based simulation) ──────────
        if self._zf_key:
            try:
                result = await self._fetch_zafronix(home_team, away_team, match_date)
                if result:
                    _logger.info("Lineup from Zafronix for %s", match_key)
                    return result
            except LineupNotAvailable:
                _logger.info("Zafronix: insufficient roster data for %s", match_key)
            except Exception as e:
                _logger.warning("Zafronix failed for %s: %s", match_key, e)

        # ── 3. player_db fallback handled by pipeline/lineup_fetcher ──
        _logger.info("All remote sources failed for %s — will use player_db", match_key)
        return None

    # ------------------------------------------------------------------
    # Flashscore scraper (lazy import to avoid hard dependency)
    # ------------------------------------------------------------------

    async def _fetch_flashscore(
        self, home_team: str, away_team: str, match_date: str,
    ) -> dict | None:
        """Fetch lineups from Flashscore's internal JSON API."""
        try:
            from services.flashscore_scraper import (
                FlashscoreError,
                get_flashscore_scraper,
            )
        except ImportError:
            _logger.debug("flashscore_scraper not available")
            return None

        try:
            scraper = get_flashscore_scraper()
            result = await scraper.fetch_lineup(home_team, away_team, match_date)
            if result is None:
                raise LineupNotAvailable("Flashscore: no lineup data returned")
            return result
        except FlashscoreError as e:
            raise LineupAPIError(f"Flashscore error: {e}") from e

    # ------------------------------------------------------------------
    # Zafronix API (lazy import)
    # ------------------------------------------------------------------

    async def _fetch_zafronix(
        self, home_team: str, away_team: str, match_date: str,
    ) -> dict | None:
        """Fetch roster-based lineups from Zafronix."""
        try:
            from services.zafronix_api import (
                ZafronixError,
                get_zafronix_client,
            )
        except ImportError:
            _logger.debug("zafronix_api not available")
            return None

        try:
            client = get_zafronix_client()
            result = await client.fetch_lineup_simulation(
                home_team, away_team, match_date,
            )
            if result is None:
                raise LineupNotAvailable("Zafronix: no roster data")
            return result
        except ZafronixError as e:
            raise LineupAPIError(f"Zafronix error: {e}") from e


# ---------------------------------------------------------------------------
# Helpers (shared)
# ---------------------------------------------------------------------------

def _fuzzy_match(a: str, b: str) -> bool:
    """Case-insensitive, diacritic-insensitive team name comparison."""
    a_norm = re.sub(r"[^a-z]", "", a.lower())
    b_norm = re.sub(r"[^a-z]", "", b.lower())
    if a_norm == b_norm:
        return True
    aliases: dict[str, list[str]] = {
        "unitedstates": ["usa", "usmnt"],
        "southkorea": ["korearepublic"],
        "czechia": ["czechrepublic"],
        "capecverde": ["capeverde"],
        "congodr": ["drcongo"],
        "bosniaherzegovina": ["bosnia"],
        "ivorycoast": ["cotedivoire"],
    }
    for canonical, aka_list in aliases.items():
        if a_norm == canonical:
            return b_norm in aka_list or b_norm == canonical
        if b_norm == canonical:
            return a_norm in aka_list or a_norm == canonical
    return a_norm in b_norm or b_norm in a_norm


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: LineupAPIClient | None = None


def get_lineup_api_client() -> LineupAPIClient:
    global _client
    if _client is None:
        _client = LineupAPIClient()
    return _client
