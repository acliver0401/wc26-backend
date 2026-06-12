"""
Flashscore lineup scraper — v1.0.0.

Scrapes confirmed starting XIs from Flashscore's internal JSON API
(free, no API key required).  Uses aiohttp with browser-mimicking headers.

Flashscore serves data through XHR endpoints under /x/feed/.  This module
implements the two-step protocol:
  1. Fetch the tournament page → extract the internal stage ID / hash
  2. Request match detail data including lineups (LU) from the feed

Usage:
    scraper = FlashscoreScraper()
    lineup = await scraper.fetch_lineup("Mexico", "South Africa", "2026-06-11")
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp

_logger = logging.getLogger("flashscore_scraper")

BASE_URL = "https://www.flashscore.com"
FEED_URL = f"{BASE_URL}/x/feed/"
MOBILE_API = "https://46.flashscore.ninja/46/x/feed/"

# World Cup 2026 tournament identifiers on Flashscore
WC26_CATEGORY = "football"
WC26_TOURNAMENT_SLUG = "world/world-cup-2026"
WC26_STAGE_ID = "jGl77cP2"  # known stage ID for WC26 group stage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Referer": f"{BASE_URL}/football/world/world-cup-2026/",
    "X-Requested-With": "XMLHttpRequest",
    "X-Fsign": "SW9D1eZo",  # common Flashscore API version header
}

MOBILE_HEADERS = {
    "User-Agent": "Android/4.4.2 Dalvik/1.6.0 (Linux; Android 14; Pixel 8 Pro)",
    "Accept": "application/json",
    "X-Fsign": "SW9D1eZo",
}


class FlashscoreError(Exception):
    """Raised when Flashscore scraping fails unrecoverably."""


class FlashscoreScraper:
    """Async scraper for Flashscore lineups via internal JSON API."""

    def __init__(
        self,
        *,
        request_timeout: int = 15,
        max_retries: int = 2,
    ):
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._match_cache: dict[str, str] = self._load_match_cache()
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_lineup(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
    ) -> dict | None:
        """
        Fetch confirmed starting XIs from Flashscore.

        Returns standardized dict:
          {home_lineup: {formation, players: [{name, pos, number}]},
           away_lineup: {formation, players: [{name, pos, number}]},
           status: "Live-Lineup", source: "flashscore"}

        Returns None if lineups not yet available.
        """
        match_key = f"{match_date}_{home_team}_{away_team}"

        # Resolve match ID
        flashscore_match_id = await self._resolve_match_id(
            home_team, away_team, match_date
        )
        if not flashscore_match_id:
            _logger.info("Flashscore: no match ID found for %s", match_key)
            return None

        # Fetch match detail (includes lineup data when available)
        raw = await self._fetch_match_detail(flashscore_match_id)
        if not raw:
            return None

        # Parse lineups from Flashscore's internal format
        home_data = raw.get("LU", {}).get("H", {})
        away_data = raw.get("LU", {}).get("A", {})

        if not home_data or not away_data:
            # Lineups might be under a different key in older API versions
            home_data = raw.get("home", {}).get("lineup", {})
            away_data = raw.get("away", {}).get("lineup", {})
            if not home_data or not away_data:
                _logger.debug("Flashscore: no lineup data for %s", match_key)
                return None

        home_formation = home_data.get("FM", home_data.get("formation", "4-4-2"))
        away_formation = away_data.get("FM", away_data.get("formation", "4-4-2"))
        home_players = self._parse_flashscore_players(
            home_data.get("PS", home_data.get("players", []))
        )
        away_players = self._parse_flashscore_players(
            away_data.get("PS", away_data.get("players", []))
        )

        if len(home_players) < 11 or len(away_players) < 11:
            _logger.debug(
                "Flashscore: incomplete lineup for %s (home:%d, away:%d)",
                match_key, len(home_players), len(away_players),
            )
            return None

        return {
            "status": "Live-Lineup",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "flashscore",
            "home_lineup": {
                "formation": home_formation,
                "players": home_players,
            },
            "away_lineup": {
                "formation": away_formation,
                "players": away_players,
            },
        }

    # ------------------------------------------------------------------
    # Match ID resolution
    # ------------------------------------------------------------------

    async def _resolve_match_id(
        self, home_team: str, away_team: str, match_date: str,
    ) -> str | None:
        """Find the Flashscore internal match ID for a fixture."""
        match_key = f"{match_date}_{home_team}_{away_team}"

        # Check cache
        if match_key in self._match_cache:
            return self._match_cache[match_key]

        # Fetch tournament stage data
        stage_data = await self._fetch_stage_data(WC26_STAGE_ID)
        if not stage_data:
            return None

        events = stage_data.get("E", stage_data.get("events", []))
        for evt in events:
            evt_date = evt.get("DT", evt.get("date", ""))
            # Flashscore date format: "11.06.2026" → "2026-06-11"
            date_normalized = _normalize_flashscore_date(evt_date)
            if date_normalized != match_date:
                continue

            h_name = evt.get("HN", evt.get("homeName", ""))
            a_name = evt.get("AN", evt.get("awayName", ""))
            if _team_match(h_name, home_team) and _team_match(a_name, away_team):
                match_id = str(evt.get("I", evt.get("id", "")))
                if match_id:
                    # Find the feed match ID (sometimes different from display ID)
                    feed_id = evt.get("FI", evt.get("feedId", match_id))
                    self._match_cache[match_key] = str(feed_id)
                    self._save_match_cache()
                    _logger.info(
                        "Flashscore: resolved match %s → ID %s", match_key, feed_id,
                    )
                    return str(feed_id)

        return None

    async def _fetch_stage_data(self, stage_id: str) -> dict | None:
        """Fetch tournament stage data from Flashscore feed."""
        # Try mobile API first (simpler, less bot protection)
        for base, headers in [
            (MOBILE_API, MOBILE_HEADERS),
            (FEED_URL, HEADERS),
        ]:
            try:
                url = f"{base}tr_1_0_{stage_id}_1_en_1"
                data = await self._get_json(url, headers)
                if data and data.get("E"):
                    return data
            except Exception:
                continue

        return None

    async def _fetch_match_detail(self, match_id: str) -> dict | None:
        """Fetch match detail JSON including lineups."""
        for base, headers in [
            (MOBILE_API, MOBILE_HEADERS),
            (FEED_URL, HEADERS),
        ]:
            try:
                url = f"{base}df_{match_id}_1"
                data = await self._get_json(url, headers)
                if data:
                    return data
            except Exception:
                continue

        return None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get_json(self, url: str, headers: dict) -> dict | None:
        """GET JSON from Flashscore with retry + error handling."""
        last_err: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as session:
                    async with session.get(url, headers=headers) as resp:
                        if resp.status == 403:
                            _logger.warning(
                                "Flashscore returned 403 for %s (bot detection)", url,
                            )
                            return None
                        if resp.status != 200:
                            body = await resp.text()
                            _logger.warning(
                                "Flashscore HTTP %d for %s: %s",
                                resp.status, url, body[:200],
                            )
                            return None

                        text = await resp.text()
                        # Flashscore wraps JSON in a callback sometimes
                        # or uses a prefix like "cricketData(" ... ")"
                        data = self._parse_flashscore_response(text)
                        return data

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    _logger.debug(
                        "Flashscore request failed (attempt %d/%d); retrying in %ds",
                        attempt + 1, self._max_retries + 1, wait,
                    )
                    await asyncio.sleep(wait)

        _logger.warning(
            "Flashscore unreachable after %d attempts: %s",
            self._max_retries + 1, last_err,
        )
        return None

    @staticmethod
    def _parse_flashscore_response(text: str) -> dict | None:
        """Parse Flashscore's JSON response, handling various wrappers."""
        text = text.strip()
        if not text:
            return None

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Some Flashscore endpoints wrap JSON in a separator-based format
        # Format: "SEP‡json_data‡SEP"
        parts = text.split("‡")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

        _logger.debug("Flashscore: could not parse response: %s", text[:300])
        return None

    # ------------------------------------------------------------------
    # Player parsing
    # ------------------------------------------------------------------

    def _parse_flashscore_players(self, players: list[dict]) -> list[dict]:
        """Parse Flashscore player objects → standard schema."""
        result: list[dict] = []
        for i, p in enumerate(players):
            # Flashscore player fields:
            #   Nm / name   → player name
            #   Pn / number → jersey number
            #   Po / pos    → position code
            #   I  / id     → player ID
            if isinstance(p, dict):
                name = (
                    p.get("Nm")
                    or p.get("name")
                    or p.get("FN", "")
                )
                number = p.get("Pn") or p.get("number") or p.get("JN", 0)
                pos = p.get("Po") or p.get("pos") or p.get("BP", "")
            else:
                name = str(p)
                number = 0
                pos = ""

            result.append({
                "name": name.strip(),
                "pos": _normalize_flashscore_pos(pos),
                "number": int(number) if number else 0,
                "rating": 80,  # backfilled later from player_db
            })

        return result

    # ------------------------------------------------------------------
    # Match ID cache
    # ------------------------------------------------------------------

    def _load_match_cache(self) -> dict[str, str]:
        path = DATA_DIR / "flashscore_match_cache.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_match_cache(self) -> None:
        path = DATA_DIR / "flashscore_match_cache.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._match_cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_flashscore_date(raw: str) -> str:
    """Convert "11.06.2026" → "2026-06-11"."""
    parts = raw.strip().split(".")
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    # Try ISO format directly
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw
    return raw


def _team_match(flashscore_name: str, our_name: str) -> bool:
    """Fuzzy team name match between Flashscore and our data."""
    if not flashscore_name or not our_name:
        return False

    a = re.sub(r"[^a-z]", "", flashscore_name.lower())
    b = re.sub(r"[^a-z]", "", our_name.lower())
    if a == b:
        return True
    if a in b or b in a:
        return True

    # Common name variations
    aliases = {
        "unitedstates": ["usa", "usmnt", "unitedstatesofamerica"],
        "southkorea": ["korearepublic", "republicofkorea"],
        "northkorea": ["koreadpr", "dprkorea"],
        "czechia": ["czechrepublic"],
        "ivorycoast": ["cotedivoire"],
        "capecverde": ["capeverde", "capeverdeislands"],
    }
    for canonical, aka_list in aliases.items():
        if a == canonical and b in aka_list:
            return True
        if b == canonical and a in aka_list:
            return True

    return False


def _normalize_flashscore_pos(raw: str) -> str:
    """Map Flashscore position codes → our standard 2-3 char format."""
    if not raw:
        return "CM"
    raw = raw.upper().strip()
    mapping = {
        "G": "GK", "GK": "GK", "GOALKEEPER": "GK",
        "D": "CB", "DF": "CB", "DC": "CB", "DEFENDER": "CB",
        "RB": "RB", "LB": "LB", "RWB": "RB", "LWB": "LB",
        "M": "CM", "MF": "CM", "MC": "CM", "MIDFIELDER": "CM",
        "DM": "DM", "CM": "CM", "AM": "AM",
        "RM": "RW", "LM": "LW",
        "F": "ST", "FW": "ST", "CF": "ST", "FORWARD": "ST",
        "RW": "RW", "LW": "LW", "RF": "RW", "LF": "LW",
        "SS": "SS", "ATT": "ST",
    }
    return mapping.get(raw, raw[:3] if len(raw) >= 2 else "CM")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_scraper: FlashscoreScraper | None = None


def get_flashscore_scraper() -> FlashscoreScraper:
    global _scraper
    if _scraper is None:
        _scraper = FlashscoreScraper()
    return _scraper
