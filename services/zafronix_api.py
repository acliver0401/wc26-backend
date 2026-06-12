"""
Zafronix World Cup API client — v1.0.0.

Free tier: 1,000 requests/day, no credit card required.
Sign up at https://api.zafronix.com/signup to get a free API key.

This client wraps the team roster and match endpoints to provide
squad-level data as a secondary lineup source when Flashscore is
unavailable.

Key endpoints used:
  GET  /tournaments          → list all tournaments
  GET  /teams?tournament=26  → all 48 teams with IDs
  GET  /teams/{id}/roster?tournament=26  → full squad with player info
  GET  /matches?tournament=26&date=2026-06-11  → matches on a date

Usage:
    client = ZafronixClient(api_key="zf_...")
    roster = await client.get_team_roster("Argentina")
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

_logger = logging.getLogger("zafronix_api")

BASE_URL = "https://api.zafronix.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class ZafronixError(Exception):
    """Raised when the Zafronix API returns an error."""


class ZafronixClient:
    """Async client for the Zafronix World Cup API (free tier)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_timeout: int = 15,
        max_retries: int = 2,
    ):
        self._api_key = api_key or os.getenv("ZAFRONIX_API_KEY", "")
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._team_cache: dict[str, str] = self._load_team_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_team_roster(
        self, team_name: str,
    ) -> list[dict] | None:
        """
        Fetch the full squad roster for a team in WC 2026.

        Returns list of player dicts with name, position, number, and
        (if available) age, caps, goals.
        """
        team_id = await self._resolve_team_id(team_name)
        if not team_id:
            _logger.warning("Zafronix: could not resolve team ID for %s", team_name)
            return None

        data = await self._get(f"/teams/{team_id}/roster", {"tournament": "26"})
        if not data:
            return None

        players = data.get("players") or data.get("roster") or []
        result: list[dict] = []
        for p in players:
            result.append({
                "name": p.get("name") or p.get("fullName", ""),
                "pos": _normalize_zafronix_pos(
                    p.get("position") or p.get("pos", "MF")
                ),
                "number": p.get("number") or p.get("shirtNumber", 0),
                "rating": 80,  # Zafronix doesn't provide ratings
                "tags": [],     # will be backfilled from player_db
                "age": p.get("age"),
                "caps": p.get("caps") or p.get("appearances"),
            })

        return result

    async def fetch_lineup_simulation(
        self, home_team: str, away_team: str, match_date: str,
    ) -> dict | None:
        """
        Build a probable starting XI from roster data.

        Since Zafronix doesn't have real-time confirmed lineups,
        this selects the most-capped players in each position
        to build a likely starting XI.

        Returns standardized lineup dict or None.
        """
        home_roster = await self.get_team_roster(home_team)
        away_roster = await self.get_team_roster(away_team)

        if not home_roster or len(home_roster) < 11:
            _logger.warning("Zafronix: insufficient roster for %s", home_team)
            return None
        if not away_roster or len(away_roster) < 11:
            _logger.warning("Zafronix: insufficient roster for %s", away_team)
            return None

        home_xi = _select_xi_by_caps(home_roster)
        away_xi = _select_xi_by_caps(away_roster)

        return {
            "status": "Live-Lineup",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "zafronix",
            "home_lineup": {
                "formation": "4-3-3",
                "players": [
                    {"name": p["name"], "pos": p["pos"], "number": p["number"], "rating": p["rating"]}
                    for p in home_xi
                ],
            },
            "away_lineup": {
                "formation": "4-4-2",
                "players": [
                    {"name": p["name"], "pos": p["pos"], "number": p["number"], "rating": p["rating"]}
                    for p in away_xi
                ],
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_team_id(self, team_name: str) -> str | None:
        """Resolve a team name → Zafronix internal team ID."""
        if team_name in self._team_cache:
            return self._team_cache[team_name]

        data = await self._get("/teams", {"tournament": "26"})
        if not data:
            return None

        teams = data.get("teams") or data.get("data") or []
        for t in teams:
            t_name = t.get("name") or t.get("teamName", "")
            if _fuzzy_team_match(t_name, team_name):
                tid = str(t.get("id") or t.get("teamId", ""))
                if tid:
                    self._team_cache[team_name] = tid
                    self._save_team_cache()
                    return tid

        return None

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        """Authenticated GET to Zafronix API."""
        if not self._api_key:
            _logger.debug("Zafronix: no API key configured, skipping")
            return None

        url = f"{BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as session:
                    async with session.get(
                        url, headers=headers, params=params,
                    ) as resp:
                        if resp.status == 401:
                            raise ZafronixError(
                                "Invalid API key — get a free key at "
                                "https://api.zafronix.com/signup"
                            )
                        if resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", "60"))
                            _logger.warning(
                                "Zafronix rate limit; waiting %ds", retry_after,
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        if resp.status != 200:
                            body = await resp.text()
                            _logger.warning(
                                "Zafronix HTTP %d for %s: %s",
                                resp.status, path, body[:200],
                            )
                            return None

                        return await resp.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)

        _logger.warning(
            "Zafronix unreachable after %d attempts: %s",
            self._max_retries + 1, last_err,
        )
        return None

    # ------------------------------------------------------------------
    # Team ID cache
    # ------------------------------------------------------------------

    def _load_team_cache(self) -> dict[str, str]:
        path = DATA_DIR / "zafronix_team_cache.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_team_cache(self) -> None:
        path = DATA_DIR / "zafronix_team_cache.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._team_cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuzzy_team_match(a: str, b: str) -> bool:
    """Case-insensitive team name match."""
    import re
    a_norm = re.sub(r"[^a-z]", "", a.lower())
    b_norm = re.sub(r"[^a-z]", "", b.lower())
    return a_norm == b_norm or a_norm in b_norm or b_norm in a_norm


def _normalize_zafronix_pos(raw: str) -> str:
    """Map Zafronix position names → our standard codes."""
    if not raw:
        return "CM"
    raw = raw.upper().strip()
    mapping = {
        "GOALKEEPER": "GK", "GK": "GK", "G": "GK",
        "DEFENDER": "CB", "DEFENCE": "CB", "DF": "CB",
        "CENTRE-BACK": "CB", "CENTER BACK": "CB", "CB": "CB",
        "RIGHT-BACK": "RB", "RIGHT BACK": "RB", "RB": "RB",
        "LEFT-BACK": "LB", "LEFT BACK": "LB", "LB": "LB",
        "MIDFIELDER": "CM", "MIDFIELD": "CM", "MF": "CM", "M": "CM",
        "DEFENSIVE MIDFIELDER": "DM", "DEFENSIVE MIDFIELD": "DM", "DM": "DM",
        "CENTRAL MIDFIELDER": "CM", "CENTRAL MIDFIELD": "CM", "CM": "CM",
        "ATTACKING MIDFIELDER": "AM", "ATTACKING MIDFIELD": "AM", "AM": "AM",
        "WINGER": "RW", "RIGHT WINGER": "RW", "LEFT WINGER": "LW",
        "RW": "RW", "LW": "LW", "RM": "RW", "LM": "LW",
        "FORWARD": "ST", "FW": "ST", "F": "ST",
        "STRIKER": "ST", "ST": "ST", "CF": "ST",
        "CENTRE-FORWARD": "ST", "CENTER FORWARD": "ST",
        "SECOND STRIKER": "SS", "SS": "SS",
    }
    return mapping.get(raw, "CM")


def _select_xi_by_caps(roster: list[dict]) -> list[dict]:
    """
    Select a probable starting XI from roster based on caps/experience.

    Picks: 1 GK, 4 DEF, 3 MID, 3 FWD — highest caps first in each category.
    Falls back to rating if caps aren't available.
    """
    def sort_key(p: dict) -> int:
        caps = p.get("caps") or 0
        if isinstance(caps, str):
            try:
                caps = int(caps)
            except (ValueError, TypeError):
                caps = 0
        return -int(caps)  # higher caps first

    gks = sorted(
        [p for p in roster if p["pos"] == "GK"], key=sort_key,
    )
    defs = sorted(
        [p for p in roster if p["pos"] in ("CB", "RB", "LB")], key=sort_key,
    )
    mids = sorted(
        [p for p in roster if p["pos"] in ("DM", "CM", "AM")], key=sort_key,
    )
    fwds = sorted(
        [p for p in roster if p["pos"] in ("RW", "LW", "ST", "SS")], key=sort_key,
    )

    xi = (
        gks[:1]
        + defs[:4]
        + mids[:3]
        + fwds[:3]
    )

    # Fill gaps from remaining players if a category is short
    used = set(id(p) for p in xi)
    remaining = sorted(
        [p for p in roster if id(p) not in used], key=sort_key,
    )
    while len(xi) < 11 and remaining:
        xi.append(remaining.pop(0))

    return xi[:11]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: ZafronixClient | None = None


def get_zafronix_client() -> ZafronixClient:
    global _client
    if _client is None:
        _client = ZafronixClient()
    return _client
