"""
API-SPORTS / Football-Data lineup client — v4.1.0.

Primary:   api-football.com v3  (x-apisports-key)
Secondary: dongqiudi.com        (reverse-engineered web API)

Usage:
    client = LineupAPIClient(api_key=os.getenv("API_FOOTBALL_KEY"))
    lineup = await client.fetch_lineup(home_team="Mexico", away_team="South Africa",
                                        match_date="2026-06-11")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import aiohttp

_logger = logging.getLogger("lineup_api")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Rate-limit & retry config
# ---------------------------------------------------------------------------

API_SPORTS_BASE = "https://v3.football.api-sports.io"
API_SPORTS_DAILY_LIMIT = 100            # free tier
DONGQIUDI_BASE = "https://dongqiudi.com/api/v3"

# ---------------------------------------------------------------------------
# Raw API response schema (api-football v3 /fixtures/lineups)
# ---------------------------------------------------------------------------
#
# GET /fixtures/lineups?fixture={id}
#
# {
#   "get": "fixtures/lineups",
#   "parameters": {"fixture": "123456"},
#   "results": 2,
#   "response": [
#     {
#       "team": {"id": 16, "name": "Mexico", "logo": "..."},
#       "formation": "4-3-3",
#       "startXI": [
#         {
#           "player": {"id": 288, "name": "R. Rangel", "number": 1, "pos": "G",
#                      "grid": "1:1"}
#         },
#         ...
#       ],
#       "substitutes": [...],
#       "coach": {"id": ..., "name": "..."}
#     },
#     { ... away team ... }
#   ]
# }
#
# Fixture search:
# GET /fixtures?date=2026-06-11&team=16
# GET /teams?search=Mexico
# ---------------------------------------------------------------------------


class LineupAPIError(Exception):
    """Raised when the lineup API returns an unrecoverable error."""


class LineupNotAvailable(Exception):
    """Raised when lineups haven't been published yet by the official source."""


class LineupAPIClient:
    """
    Multi-source lineup client with automatic fallback.

    Priority:
      1. api-football.com v3  (authoritative, 100 calls/day free)
      2. Dongqiudi web API   (Chinese platform, good WC coverage)
      3. Local fixture-id cache to avoid redundant fixture lookups
    """

    def __init__(
        self,
        api_football_key: str | None = None,
        dongqiudi_token: str | None = None,
        *,
        request_timeout: int = 15,
        max_retries: int = 2,
    ):
        self._api_key = api_football_key or os.getenv("API_FOOTBALL_KEY", "")
        self._dq_token = dongqiudi_token or os.getenv("DONGQIUDI_TOKEN", "")
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._fixture_id_cache: dict[str, int] = self._load_fixture_cache()
        self._team_id_cache: dict[str, int] = self._load_team_id_cache()

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

        Returns a dict with keys:
          home_lineup:  {formation, players: [{name, pos, number, rating}]}
          away_lineup:  {formation, players: [{name, pos, number, rating}]}
          status:       "Live-Lineup"
          fetched_at:   ISO timestamp
          source:       "api-football" | "dongqiudi"

        Returns None if lineups are not yet available.
        Raises LineupAPIError on network/auth failures.
        """
        match_key = f"{match_date}_{home_team}_{away_team}"

        # 1 — Try API-SPORTS
        if self._api_key:
            try:
                result = await self._fetch_api_sports(home_team, away_team, match_date)
                if result:
                    _logger.info("Lineup from api-football for %s", match_key)
                    return result
            except LineupNotAvailable:
                _logger.info("api-football: lineups not yet available for %s", match_key)
            except LineupAPIError as e:
                _logger.warning("api-football failed for %s: %s", match_key, e)

        # 2 — Try Dongqiudi (懂球帝)
        if self._dq_token:
            try:
                result = await self._fetch_dongqiudi(home_team, away_team, match_date)
                if result:
                    _logger.info("Lineup from dongqiudi for %s", match_key)
                    return result
            except LineupNotAvailable:
                _logger.info("dongqiudi: lineups not yet available for %s", match_key)
            except LineupAPIError as e:
                _logger.warning("dongqiudi failed for %s: %s", match_key, e)

        return None

    # ------------------------------------------------------------------
    # API-SPORTS v3  (api-football.com)
    # ------------------------------------------------------------------

    async def _fetch_api_sports(
        self, home_team: str, away_team: str, match_date: str,
    ) -> dict | None:
        """Full pipeline: resolve fixture ID → fetch lineups → validate."""
        fixture_id = await self._resolve_fixture_id(home_team, away_team, match_date)
        if not fixture_id:
            raise LineupNotAvailable(
                f"No fixture ID found for {home_team} vs {away_team} on {match_date}"
            )

        raw = await self._api_sports_get("/fixtures/lineups", {"fixture": str(fixture_id)})
        responses = raw.get("response", [])

        if not responses or len(responses) < 2:
            raise LineupNotAvailable(
                f"Fixture {fixture_id}: API returned {len(responses)} team lineups (expected 2)"
            )

        home_data, away_data = self._assign_home_away(responses, home_team, away_team)

        # --- Validate before returning ---
        self._validate_lineup_segment(home_data, home_team)
        self._validate_lineup_segment(away_data, away_team)

        home_players = self._parse_start_xi(home_data["startXI"])
        away_players = self._parse_start_xi(away_data["startXI"])

        return {
            "status": "Live-Lineup",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "api-football",
            "fixture_id": fixture_id,
            "home_lineup": {
                "formation": home_data.get("formation", "4-4-2"),
                "players": home_players,
                "coach": home_data.get("coach", {}).get("name", ""),
            },
            "away_lineup": {
                "formation": away_data.get("formation", "4-4-2"),
                "players": away_players,
                "coach": away_data.get("coach", {}).get("name", ""),
            },
        }

    async def _resolve_fixture_id(
        self, home_team: str, away_team: str, match_date: str,
    ) -> int | None:
        """
        Map (date, home_team, away_team) → api-football fixture ID.

        Uses a local JSON cache to avoid redundant API calls.
        """
        cache_key = f"{match_date}_{home_team}_{away_team}"

        # Check in-memory cache
        if cache_key in self._fixture_id_cache:
            return self._fixture_id_cache[cache_key]

        # Resolve team IDs
        home_id = await self._resolve_team_id(home_team)
        away_id = await self._resolve_team_id(away_team)
        if not home_id or not away_id:
            return None

        # Query fixtures for that date, filtered by home team
        params = {"date": match_date, "team": str(home_id)}
        raw = await self._api_sports_get("/fixtures", params)

        for fx in raw.get("response", []):
            teams = fx.get("teams", {})
            h_name = teams.get("home", {}).get("name", "")
            a_name = teams.get("away", {}).get("name", "")
            if _fuzzy_match(h_name, home_team) and _fuzzy_match(a_name, away_team):
                fid = fx["fixture"]["id"]
                self._fixture_id_cache[cache_key] = fid
                self._save_fixture_cache()
                _logger.info("Resolved fixture %d for %s", fid, cache_key)
                return fid

        _logger.warning("No fixture match for %s vs %s on %s", home_team, away_team, match_date)
        return None

    async def _resolve_team_id(self, team_name: str) -> int | None:
        """Resolve team name → api-football team ID."""
        if team_name in self._team_id_cache:
            return self._team_id_cache[team_name]

        raw = await self._api_sports_get("/teams", {"search": team_name})
        for t in raw.get("response", []):
            t_name = t.get("team", {}).get("name", "")
            if _fuzzy_match(t_name, team_name):
                tid = t["team"]["id"]
                self._team_id_cache[team_name] = tid
                self._save_team_id_cache()
                return tid

        # Try country name lookup
        raw2 = await self._api_sports_get("/teams", {"country": team_name})
        for t in raw2.get("response", []):
            if t.get("team", {}).get("national"):
                tid = t["team"]["id"]
                self._team_id_cache[team_name] = tid
                self._save_team_id_cache()
                return tid

        return None

    async def _api_sports_get(self, path: str, params: dict) -> dict:
        """Authenticated GET to api-football.com v3 with retry + rate-limit."""
        url = f"{API_SPORTS_BASE}{path}"
        headers = {
            "x-apisports-key": self._api_key,
            "Accept": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout)) as session:
                    async with session.get(url, headers=headers, params=params) as resp:
                        if resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", "60"))
                            _logger.warning("API-SPORTS rate limit; waiting %ds", retry_after)
                            await asyncio.sleep(retry_after)
                            continue

                        if resp.status == 404:
                            raise LineupNotAvailable(f"404 from {path}")

                        if resp.status != 200:
                            body = await resp.text()
                            raise LineupAPIError(
                                f"api-football {path} returned {resp.status}: {body[:300]}"
                            )

                        data = await resp.json()
                        errors = data.get("errors", [])
                        if errors:
                            err_msg = "; ".join(str(e) for e in errors)
                            if "rate limit" in err_msg.lower():
                                _logger.warning("API rate limit: %s", err_msg)
                                await asyncio.sleep(60)
                                continue
                            raise LineupAPIError(f"api-football errors: {err_msg}")

                        # Check remaining quota
                        remaining = resp.headers.get("x-ratelimit-requests-remaining")
                        if remaining and int(remaining) < 10:
                            _logger.warning("API-SPORTS quota low: %s remaining", remaining)

                        return data

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    _logger.warning("API-SPORTS request failed (attempt %d/%d); retrying in %ds",
                                    attempt + 1, self._max_retries + 1, wait)
                    await asyncio.sleep(wait)

        raise LineupAPIError(f"api-football unreachable after {self._max_retries + 1} attempts: {last_err}")

    # ------------------------------------------------------------------
    # Dongqiudi (懂球帝) API
    # ------------------------------------------------------------------

    async def _fetch_dongqiudi(
        self, home_team: str, away_team: str, match_date: str,
    ) -> dict | None:
        """
        Fetch lineups from Dongqiudi's web API.

        Endpoint (reverse-engineered):
          GET /api/v3/archive/match/{match_id}/lineup

        Headers required:
          Authorization: Bearer {token}
          User-Agent: Dongqiudi/{version} (Android)
        """
        match_id = await self._dq_resolve_match_id(home_team, away_team, match_date)
        if not match_id:
            raise LineupNotAvailable("dongqiudi: could not resolve match ID")

        headers = {
            "Authorization": f"Bearer {self._dq_token}",
            "User-Agent": "Dongqiudi/7.8.0 (Android 14; Scale/3.0)",
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
        }

        url = f"{DONGQIUDI_BASE}/archive/match/{match_id}/lineup"
        last_err: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout)) as session:
                    async with session.get(url, headers=headers) as resp:
                        if resp.status == 404:
                            raise LineupNotAvailable("dongqiudi: match not found")
                        if resp.status != 200:
                            raise LineupAPIError(f"dongqiudi returned {resp.status}")

                        data = await resp.json()

                        # Dongqiudi wraps lineups in data.content.lineup
                        content = data.get("data", {}).get("content", {})
                        lineup = content.get("lineup") or data.get("data", {}).get("lineup", {})

                        if not lineup or not lineup.get("home") or not lineup.get("away"):
                            raise LineupNotAvailable("dongqiudi: lineup not yet published")

                        home_raw = lineup["home"]
                        away_raw = lineup["away"]

                        self._validate_lineup_raw(home_raw, home_team)
                        self._validate_lineup_raw(away_raw, away_team)

                        return self._dq_parse_lineup(home_raw, away_raw, home_team, away_team)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)

        raise LineupAPIError(f"dongqiudi unreachable: {last_err}")

    async def _dq_resolve_match_id(
        self, home_team: str, away_team: str, match_date: str,
    ) -> str | None:
        """Resolve a match ID from Dongqiudi's schedule endpoint."""
        headers = {
            "Authorization": f"Bearer {self._dq_token}",
            "User-Agent": "Dongqiudi/7.8.0 (Android)",
            "Accept": "application/json",
        }

        url = f"{DONGQIUDI_BASE}/schedule/list"
        params = {"date": match_date}

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout)) as session:
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
        except Exception:
            return None

        matches = data.get("data", {}).get("matches", [])
        for m in matches:
            h = m.get("home_team_name", "")
            a = m.get("away_team_name", "")
            if _fuzzy_match(h, home_team) and _fuzzy_match(a, away_team):
                return str(m.get("id") or m.get("match_id", ""))

        return None

    def _dq_parse_lineup(
        self, home_raw: dict, away_raw: dict, home_team: str, away_team: str,
    ) -> dict:
        """Parse Dongqiudi's lineup JSON into our standard schema."""
        home_players = self._dq_parse_players(home_raw.get("players") or home_raw.get("startXI", []))
        away_players = self._dq_parse_players(away_raw.get("players") or away_raw.get("startXI", []))

        return {
            "status": "Live-Lineup",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "dongqiudi",
            "home_lineup": {
                "formation": home_raw.get("formation", "4-4-2"),
                "players": home_players,
            },
            "away_lineup": {
                "formation": away_raw.get("formation", "4-4-2"),
                "players": away_players,
            },
        }

    def _dq_parse_players(self, players: list[dict]) -> list[dict]:
        """Parse Dongqiudi player objects → standard schema."""
        result: list[dict] = []
        for i, p in enumerate(players):
            # Dongqiudi wraps player info differently depending on API version
            if isinstance(p, dict):
                info = p.get("player") or p
                name = (
                    info.get("name")
                    or info.get("shown_name")
                    or info.get("person", {}).get("name")
                    or ""
                )
                number = info.get("number") or info.get("shirt_number") or info.get("num", 0)
                pos = info.get("pos") or info.get("position") or info.get("role", "")
            else:
                name = str(p)
                number = 0
                pos = ""

            result.append({
                "name": name.strip(),
                "pos": pos.upper()[:3] if pos else _infer_position(i),
                "number": int(number) if number else 0,
                "rating": 80,  # Dongqiudi doesn't provide ratings; backfilled from player_db
            })

        return result

    # ------------------------------------------------------------------
    # Validation (shared across sources)
    # ------------------------------------------------------------------

    def _assign_home_away(
        self, responses: list[dict], home_team: str, away_team: str,
    ) -> tuple[dict, dict]:
        """Assign the two response entries to home/away by fuzzy-matching team name."""
        t0 = responses[0].get("team", {}).get("name", "")
        if _fuzzy_match(t0, home_team):
            return responses[0], responses[1]
        return responses[1], responses[0]

    def _validate_lineup_segment(self, data: dict, expected_team: str) -> None:
        """Raise LineupNotAvailable if the lineup segment is invalid."""
        team_name = data.get("team", {}).get("name", "")
        start_xi = data.get("startXI", [])

        if not start_xi or len(start_xi) < 11:
            raise LineupNotAvailable(
                f"Lineup for {expected_team} has only {len(start_xi)} players (expected 11)"
            )

        if not _fuzzy_match(team_name, expected_team):
            _logger.warning(
                "Team name mismatch in lineup: expected %s, got %s", expected_team, team_name,
            )

        # Reject fake/placeholder names
        for entry in start_xi:
            player = entry.get("player", entry)
            name = player.get("name", "")
            if not name or _is_placeholder_name(name):
                raise LineupNotAvailable(
                    f"Placeholder player name detected in {expected_team}: '{name}'"
                )

    def _validate_lineup_raw(self, raw: dict, expected_team: str) -> None:
        """Validate Dongqiudi-style raw lineup data."""
        players = raw.get("players") or raw.get("startXI") or []
        if not players or len(players) < 11:
            raise LineupNotAvailable(
                f"Dongqiudi lineup for {expected_team} has only {len(players)} players"
            )

    def _parse_start_xi(self, start_xi: list[dict]) -> list[dict]:
        """Convert api-football startXI entries → standard player schema."""
        players: list[dict] = []
        for entry in start_xi:
            p = entry.get("player", entry)
            name = p.get("name", "")
            pos = p.get("pos", "")
            number = p.get("number", 0)
            players.append({
                "name": name.strip(),
                "pos": _normalize_position(pos),
                "number": int(number) if number else 0,
                # Rating backfilled from player_db after merge
                "rating": 80,
            })
        return players

    # ------------------------------------------------------------------
    # Fixture / team ID cache persistence
    # ------------------------------------------------------------------

    def _load_fixture_cache(self) -> dict[str, int]:
        path = DATA_DIR / "fixture_id_cache.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_fixture_cache(self) -> None:
        path = DATA_DIR / "fixture_id_cache.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._fixture_id_cache, f, ensure_ascii=False, indent=2)

    def _load_team_id_cache(self) -> dict[str, int]:
        path = DATA_DIR / "team_id_cache.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_team_id_cache(self) -> None:
        path = DATA_DIR / "team_id_cache.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._team_id_cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(a: str, b: str) -> bool:
    """Case-insensitive, diacritic-insensitive team name comparison."""
    a_norm = re.sub(r"[^a-z]", "", a.lower())
    b_norm = re.sub(r"[^a-z]", "", b.lower())
    if a_norm == b_norm:
        return True
    # Handle common aliases
    aliases: dict[str, list[str]] = {
        "unitedstates": ["usa", "usmnt"],
        "southkorea": ["korearepublic"],
        "czechia": ["czechrepublic"],
        "capecverde": ["capeverde", "capeverdeislands"],
        "congodr": ["drcongo", "congo"],
        "bosniaherzegovina": ["bosnia"],
        "ivorycoast": ["cotedivoire"],
    }
    for canonical, aka_list in aliases.items():
        if a_norm == canonical:
            return b_norm in aka_list or b_norm == canonical
        if b_norm == canonical:
            return a_norm in aka_list or a_norm == canonical
    # Substring match
    return a_norm in b_norm or b_norm in a_norm


def _normalize_position(raw: str) -> str:
    """Map api-football position codes to our standard 2-3 char codes."""
    raw = raw.upper().strip()
    mapping = {
        "G": "GK", "GK": "GK",
        "D": "CB", "DF": "CB", "DC": "CB",
        "RB": "RB", "LB": "LB", "RWB": "RB", "LWB": "LB",
        "M": "CM", "MF": "CM", "MC": "CM",
        "DM": "DM", "CM": "CM", "AM": "AM",
        "RM": "RW", "LM": "LW",
        "F": "ST", "FW": "ST", "CF": "ST",
        "RW": "RW", "LW": "LW", "RF": "RW", "LF": "LW",
        "SS": "SS",
    }
    return mapping.get(raw, raw[:3])


def _is_placeholder_name(name: str) -> bool:
    """Detect generic/placeholder player names that should never go to production."""
    name_lower = name.lower().strip()
    patterns = [
        r"^player\s*\d+$",           # "Player 1", "Player 11"
        r"^unknown",                   # "Unknown Player"
        r"^tba",                       # "TBA"
        r"^n/?a$",                     # "N/A"
        r"^\s*$",                      # empty
        r"^substitute\s*\d*$",         # "Substitute" / "Substitute 1"
        r"^placeholder",
    ]
    return any(re.match(p, name_lower) for p in patterns)


def _infer_position(index: int) -> str:
    """Fallback position inference by lineup index (1→GK, 2-5→DEF, 6-8→MID, 9-11→FWD)."""
    if index == 0:
        return "GK"
    if index < 5:
        return "CB"
    if index < 8:
        return "CM"
    return "ST"


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_client: LineupAPIClient | None = None


def get_lineup_api_client() -> LineupAPIClient:
    """Return the singleton LineupAPIClient, creating it on first call."""
    global _client
    if _client is None:
        _client = LineupAPIClient()
    return _client
